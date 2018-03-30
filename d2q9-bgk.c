/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

typedef struct {
  float *speed0;
  float *speed1;
  float *speed2;
  float *speed3;
  float *speed4;
  float *speed5;
  float *speed6;
  float *speed7;
  float *speed8;
} s_speeds;

typedef struct {
  float speed0;
  float speed1;
  float speed2;
  float speed3;
  float speed4;
  float speed5;
  float speed6;
  float speed7;
  float speed8;
} s_tmp_speeds;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speeds *speeds, s_speeds *tmp_speeds,
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestepInner(const t_param params, s_speeds* speeds, s_speeds* tmp_speeds, int* obstacles);
int timestepOuter(const t_param params, s_speeds* speeds, s_speeds* tmp_speeds, int* obstacles);
int accelerate_flow(const t_param params, s_speeds* speeds, int* obstacles);
int write_values(const t_param params, s_speeds* speeds, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, s_speeds* speeds, s_speeds* tmp_speeds,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, s_speeds* speeds);

float total_velocity(const t_param params, s_speeds* speeds, int* obstacles);
/* compute average velocity */
float av_velocity(const t_param params, s_speeds* speeds, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, s_speeds* speeds, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[]) {
  char* paramfile = NULL;    /* name of the input parameter file */
  char* obstaclefile = NULL; /* name of a the input obstacle file */
  t_param params;              /* struct to hold parameter values */
  s_speeds* speeds = malloc(sizeof(s_speeds));
  s_speeds* tmp_speeds = malloc(sizeof(s_speeds));
  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  int worldSize;
  int rank;

  MPI_Init(NULL, NULL);

  MPI_Datatype MPI_T_PARAM;
  MPI_Datatype types2[] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
  int blocklen2[] = {1, 1, 1, 1, 1, 1, 1};
  MPI_Aint disp2[] = {0, 1 * sizeof(int), 2 * sizeof(int), 3 * sizeof(int), 4 * sizeof(int), 4 * sizeof(float) + sizeof(float), 4 * sizeof(int) + 2 * sizeof(float)};
  MPI_Type_create_struct(7, blocklen2, disp2, types2, &MPI_T_PARAM);
  MPI_Type_commit(&MPI_T_PARAM);

  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ndims = 1;
  int dims[] = {worldSize};
  int periods[] = {1};
  MPI_Comm cart_world;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_world);

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  int tot_cells = 0;
  int cols_per_proc;
  int *send_cnts = malloc(sizeof(int) * worldSize);
  int *row_cnts = malloc(sizeof(int) * worldSize);
  int *displs = malloc(sizeof(int) * worldSize);

  /* initialise our data structures and load values from file */
  if (rank == 0) {
    initialise(paramfile, obstaclefile, &params, speeds, tmp_speeds, &obstacles, &av_vels);

    //Calc total number of non-obstacle cells
    for (int j = 0; j < params.ny; j++) {
      for (int i = 0; i < params.nx; i++) {
        tot_cells += !obstacles[i + j * params.nx];
      }
    }

    int rem = params.ny % worldSize;
    int sum = 0;
    for (int i = 0; i < worldSize; i++) {
        row_cnts[i] = params.ny / worldSize;
        if (rem > 0) {
            row_cnts[i]++;
            rem--;
        }
        send_cnts[i] = params.nx * row_cnts[i];

        displs[i] = sum;
        sum += send_cnts[i];
    }
    cols_per_proc = params.nx;
  }
  MPI_Bcast(send_cnts, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(row_cnts, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displs, worldSize, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&params, 1, MPI_T_PARAM, 0, MPI_COMM_WORLD);

  t_param sub_params;
  memcpy(&sub_params, &params, sizeof(t_param));
  sub_params.ny = row_cnts[rank];
  sub_params.nx = cols_per_proc;

  int *sub_obstacles = malloc(sizeof(int) * sub_params.nx * (sub_params.ny + 2));

  s_speeds *sub_speeds = malloc(sizeof(s_speeds));
  sub_speeds->speed0 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed1 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed2 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed3 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed4 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed5 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed6 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed7 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_speeds->speed8 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));

  s_speeds *sub_tmp_speeds = malloc(sizeof(s_speeds));
  sub_tmp_speeds->speed0 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed1 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed2 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed3 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed4 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed5 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed6 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed7 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));
  sub_tmp_speeds->speed8 = malloc(sizeof(float) * sub_params.nx * (sub_params.ny + 2));

  MPI_Scatterv(obstacles, send_cnts, displs, MPI_INT, (sub_obstacles + sub_params.nx), send_cnts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(speeds->speed0, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed1, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed2, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed3, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed4, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed5, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed6, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed7, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speeds->speed8, send_cnts, displs, MPI_FLOAT, (sub_speeds->speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Request haloRequest0, haloRequest1, haloRequest2, haloRequest3, haloRequest4, haloRequest5, haloRequest6, haloRequest7, haloRequest8;
  MPI_Status haloStatus0, haloStatus1, haloStatus2, haloStatus3, haloStatus4, haloStatus5, haloStatus6, haloStatus7, haloStatus8;
  s_speeds *swp;

  float* sendbuf0 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf1 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf2 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf3 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf4 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf5 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf6 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf7 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf8 = malloc(sizeof(float) * 2 * cols_per_proc);

  float* recvbuf0 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf1 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf2 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf3 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf4 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf5 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf6 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf7 = malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf8 = malloc(sizeof(float) * 2 * cols_per_proc);

  if (rank == 0) {
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }

  /* iterate for maxIters timesteps */
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    if ((worldSize - 1) == rank) {
      accelerate_flow(sub_params, sub_speeds, sub_obstacles);
    }

    memcpy(sendbuf0, sub_speeds->speed0 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf0 + sub_params.nx, sub_speeds->speed0 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf1, sub_speeds->speed1 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf1 + sub_params.nx, sub_speeds->speed1 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf2, sub_speeds->speed2 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf2 + sub_params.nx, sub_speeds->speed2 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf3, sub_speeds->speed3 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf3 + sub_params.nx, sub_speeds->speed3 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf4, sub_speeds->speed4 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf4 + sub_params.nx, sub_speeds->speed4 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf5, sub_speeds->speed5 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf5 + sub_params.nx, sub_speeds->speed5 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf6, sub_speeds->speed6 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf6 + sub_params.nx, sub_speeds->speed6 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf7, sub_speeds->speed7 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf7 + sub_params.nx, sub_speeds->speed7 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf8, sub_speeds->speed8 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf8 + sub_params.nx, sub_speeds->speed8 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);

    MPI_Ineighbor_alltoall(sendbuf0, sub_params.nx, MPI_FLOAT, recvbuf0, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest0);
    MPI_Ineighbor_alltoall(sendbuf1, sub_params.nx, MPI_FLOAT, recvbuf1, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest1);
    MPI_Ineighbor_alltoall(sendbuf2, sub_params.nx, MPI_FLOAT, recvbuf2, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest2);
    MPI_Ineighbor_alltoall(sendbuf3, sub_params.nx, MPI_FLOAT, recvbuf3, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest3);
    MPI_Ineighbor_alltoall(sendbuf4, sub_params.nx, MPI_FLOAT, recvbuf4, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest4);
    MPI_Ineighbor_alltoall(sendbuf5, sub_params.nx, MPI_FLOAT, recvbuf5, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest5);
    MPI_Ineighbor_alltoall(sendbuf6, sub_params.nx, MPI_FLOAT, recvbuf6, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest6);
    MPI_Ineighbor_alltoall(sendbuf7, sub_params.nx, MPI_FLOAT, recvbuf7, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest7);
    MPI_Ineighbor_alltoall(sendbuf8, sub_params.nx, MPI_FLOAT, recvbuf8, sub_params.nx, MPI_FLOAT, cart_world, &haloRequest8);

    #pragma forceinline
    timestepInner(sub_params, sub_speeds, sub_tmp_speeds, sub_obstacles);

    MPI_Wait(&haloRequest0, &haloStatus0);
    MPI_Wait(&haloRequest1, &haloStatus1);
    MPI_Wait(&haloRequest2, &haloStatus2);
    MPI_Wait(&haloRequest3, &haloStatus3);
    MPI_Wait(&haloRequest4, &haloStatus4);
    MPI_Wait(&haloRequest5, &haloStatus5);
    MPI_Wait(&haloRequest6, &haloStatus6);
    MPI_Wait(&haloRequest7, &haloStatus7);
    MPI_Wait(&haloRequest8, &haloStatus8);

    if (worldSize != 2) {
      memcpy(sub_speeds->speed0 + ((sub_params.ny + 1) * sub_params.nx), recvbuf0 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed0, recvbuf0, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed1 + ((sub_params.ny + 1) * sub_params.nx), recvbuf1 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed1, recvbuf1, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed2 + ((sub_params.ny + 1) * sub_params.nx), recvbuf2 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed2, recvbuf2, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed3 + ((sub_params.ny + 1) * sub_params.nx), recvbuf3 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed3, recvbuf3, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed4, recvbuf4, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed5 + ((sub_params.ny + 1) * sub_params.nx), recvbuf5 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed5, recvbuf5, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed6 + ((sub_params.ny + 1) * sub_params.nx), recvbuf6 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed6, recvbuf6, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed7, recvbuf7, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed8, recvbuf8, sizeof(float) * sub_params.nx);
    } else {
      memcpy(sub_speeds->speed0 + ((sub_params.ny + 1) * sub_params.nx), recvbuf0, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed0, recvbuf0 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed1 + ((sub_params.ny + 1) * sub_params.nx), recvbuf1, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed1, recvbuf1 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed2 + ((sub_params.ny + 1) * sub_params.nx), recvbuf2, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed2, recvbuf2 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed3 + ((sub_params.ny + 1) * sub_params.nx), recvbuf3, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed3, recvbuf3 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed4, recvbuf4 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed5 + ((sub_params.ny + 1) * sub_params.nx), recvbuf5, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed5, recvbuf5 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed6 + ((sub_params.ny + 1) * sub_params.nx), recvbuf6, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed6, recvbuf6 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed7, recvbuf7 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8, sizeof(float) * sub_params.nx);
      memcpy(sub_speeds->speed8, recvbuf8 + sub_params.nx, sizeof(float) * sub_params.nx);
    }

    #pragma forceinline
    timestepOuter(sub_params, sub_speeds, sub_tmp_speeds, sub_obstacles);
    swp = sub_speeds;
    sub_speeds = sub_tmp_speeds;
    sub_tmp_speeds = swp;

    float local_tot_vel = total_velocity(sub_params, sub_speeds, sub_obstacles);
    float global_tot_vel;
    MPI_Reduce(&local_tot_vel, &global_tot_vel, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      av_vels[tt] = global_tot_vel / (float) tot_cells;
    }
#ifdef DEBUG
  if (rank == 0) {
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(sub_params, sub_speeds));
  }
#endif
  }

  if (rank == 0) {
    gettimeofday(&timstr, NULL);
    toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    getrusage(RUSAGE_SELF, &ru);
    timstr = ru.ru_utime;
    usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    timstr = ru.ru_stime;
    systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }

  MPI_Gatherv((sub_obstacles + sub_params.nx), send_cnts[rank], MPI_INT, obstacles, send_cnts, displs, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Gatherv((sub_speeds->speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed0, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed1, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed2, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed3, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed4, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed5, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed6, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed7, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speeds->speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speeds->speed8, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Finalize();

  if (rank == 0) {
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, speeds, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, speeds, obstacles, av_vels);
    finalise(&params, speeds, tmp_speeds, &obstacles, &av_vels);
  }

  return EXIT_SUCCESS;
}

int timestepInner(const t_param params, s_speeds* speeds, s_speeds* tmp_speeds, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  s_tmp_speeds tmpSpeed;

  int y_n, x_e, y_s, x_w;
  float local_density, u_x, u_y, u_sq;
  /* loop over _all_ cells */
  #pragma omp simd collapse(2) private(tmpSpeed, y_n, x_e, y_s, x_w, local_density, u_x, u_y, u_sq)
  for (int jj = 2; jj < params.ny; jj++) {
    for (int ii = 0; ii < params.nx; ii++) {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      y_n = jj + 1;
      x_e = (ii + 1) % params.nx;
      y_s = jj - 1;
      x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmpSpeed.speed0 = speeds->speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmpSpeed.speed1 = speeds->speed1[x_w + jj*params.nx]; /* east */
      tmpSpeed.speed2 = speeds->speed2[ii + y_s*params.nx]; /* north */
      tmpSpeed.speed3 = speeds->speed3[x_e + jj*params.nx]; /* west */
      tmpSpeed.speed4 = speeds->speed4[ii + y_n*params.nx]; /* south */
      tmpSpeed.speed5 = speeds->speed5[x_w + y_s*params.nx]; /* north-east */
      tmpSpeed.speed6 = speeds->speed6[x_e + y_s*params.nx]; /* north-west */
      tmpSpeed.speed7 = speeds->speed7[x_e + y_n*params.nx]; /* south-west */
      tmpSpeed.speed8 = speeds->speed8[x_w + y_n*params.nx]; /* south-east */

      /* compute local density total */
      local_density = 0.f;

      local_density += tmpSpeed.speed0;
      local_density += tmpSpeed.speed1;
      local_density += tmpSpeed.speed2;
      local_density += tmpSpeed.speed3;
      local_density += tmpSpeed.speed4;
      local_density += tmpSpeed.speed5;
      local_density += tmpSpeed.speed6;
      local_density += tmpSpeed.speed7;
      local_density += tmpSpeed.speed8;

      /* compute x velocity component */
      u_x = (tmpSpeed.speed1
                    + tmpSpeed.speed5
                    + tmpSpeed.speed8
                    - (tmpSpeed.speed3
                        + tmpSpeed.speed6
                        + tmpSpeed.speed7))
                    / local_density;
      /* compute y velocity component */
      u_y = (tmpSpeed.speed2
                    + tmpSpeed.speed5
                    + tmpSpeed.speed6
                    - (tmpSpeed.speed4
                        + tmpSpeed.speed7
                        + tmpSpeed.speed8))
                    / local_density;

      /* velocity squared */
      u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

      /* relaxation step */
      tmp_speeds->speed0[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed0
                                                + params.omega
                                              * (d_equ[0] - tmpSpeed.speed0) : tmpSpeed.speed0;
      tmp_speeds->speed1[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed1
                                              + params.omega
                                              * (d_equ[1] - tmpSpeed.speed1) : tmpSpeed.speed3;
      tmp_speeds->speed2[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed2
                                              + params.omega
                                              * (d_equ[2] - tmpSpeed.speed2) : tmpSpeed.speed4;
      tmp_speeds->speed3[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed3
                                              + params.omega
                                              * (d_equ[3] - tmpSpeed.speed3) : tmpSpeed.speed1;
      tmp_speeds->speed4[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed4
                                              + params.omega
                                              * (d_equ[4] - tmpSpeed.speed4) : tmpSpeed.speed2;
      tmp_speeds->speed5[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed5
                                              + params.omega
                                              * (d_equ[5] - tmpSpeed.speed5) : tmpSpeed.speed7;
      tmp_speeds->speed6[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed6
                                              + params.omega
                                              * (d_equ[6] - tmpSpeed.speed6) : tmpSpeed.speed8;
      tmp_speeds->speed7[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed7
                                              + params.omega
                                              * (d_equ[7] - tmpSpeed.speed7) : tmpSpeed.speed5;
      tmp_speeds->speed8[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed8
                                              + params.omega
                                              * (d_equ[8] - tmpSpeed.speed8) : tmpSpeed.speed6;
    }
  }
  return EXIT_SUCCESS;
}

int timestepOuter(const t_param params, s_speeds* speeds, s_speeds* tmp_speeds, int* obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  s_tmp_speeds tmpSpeed;

  int y_n, x_e, y_s, x_w;
  float local_density, u_x, u_y, u_sq;
  int jj;
  #pragma omp simd collapse(2) private(jj ,tmpSpeed, y_n, x_e, y_s, x_w, local_density, u_x, u_y, u_sq)
  for (int c = 0; c < 2; c++) {
    for (int ii = 0; ii < params.nx; ii++) {
      jj = c == 0 ? 1 : params.ny;
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      y_n = jj + 1;
      x_e = (ii + 1) % params.nx;
      y_s = jj - 1;
      x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      /* propagate densities from neighbouring cells, following
      ** appropriate directions of travel and writing into
      ** scratch space grid */
      tmpSpeed.speed0 = speeds->speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmpSpeed.speed1 = speeds->speed1[x_w + jj*params.nx]; /* east */
      tmpSpeed.speed2 = speeds->speed2[ii + y_s*params.nx]; /* north */
      tmpSpeed.speed3 = speeds->speed3[x_e + jj*params.nx]; /* west */
      tmpSpeed.speed4 = speeds->speed4[ii + y_n*params.nx]; /* south */
      tmpSpeed.speed5 = speeds->speed5[x_w + y_s*params.nx]; /* north-east */
      tmpSpeed.speed6 = speeds->speed6[x_e + y_s*params.nx]; /* north-west */
      tmpSpeed.speed7 = speeds->speed7[x_e + y_n*params.nx]; /* south-west */
      tmpSpeed.speed8 = speeds->speed8[x_w + y_n*params.nx]; /* south-east */

      /* compute local density total */
      local_density = 0.f;

      local_density += tmpSpeed.speed0;
      local_density += tmpSpeed.speed1;
      local_density += tmpSpeed.speed2;
      local_density += tmpSpeed.speed3;
      local_density += tmpSpeed.speed4;
      local_density += tmpSpeed.speed5;
      local_density += tmpSpeed.speed6;
      local_density += tmpSpeed.speed7;
      local_density += tmpSpeed.speed8;

      /* compute x velocity component */
      u_x = (tmpSpeed.speed1
                    + tmpSpeed.speed5
                    + tmpSpeed.speed8
                    - (tmpSpeed.speed3
                        + tmpSpeed.speed6
                        + tmpSpeed.speed7))
                    / local_density;
      /* compute y velocity component */
      u_y = (tmpSpeed.speed2
                    + tmpSpeed.speed5
                    + tmpSpeed.speed6
                    - (tmpSpeed.speed4
                        + tmpSpeed.speed7
                        + tmpSpeed.speed8))
                    / local_density;

      /* velocity squared */
      u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = w0 * local_density
                  * (1.f - u_sq / (2.f * c_sq));
      /* axis speeds: weight w1 */
      d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                        + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                        + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                        + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                        + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      /* diagonal speeds: weight w2 */
      d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                        + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                        + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                        + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));
      d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                        + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                        - u_sq / (2.f * c_sq));

      /* relaxation step */
      tmp_speeds->speed0[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed0
                                                + params.omega
                                              * (d_equ[0] - tmpSpeed.speed0) : tmpSpeed.speed0;
      tmp_speeds->speed1[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed1
                                              + params.omega
                                              * (d_equ[1] - tmpSpeed.speed1) : tmpSpeed.speed3;
      tmp_speeds->speed2[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed2
                                              + params.omega
                                              * (d_equ[2] - tmpSpeed.speed2) : tmpSpeed.speed4;
      tmp_speeds->speed3[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed3
                                              + params.omega
                                              * (d_equ[3] - tmpSpeed.speed3) : tmpSpeed.speed1;
      tmp_speeds->speed4[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed4
                                              + params.omega
                                              * (d_equ[4] - tmpSpeed.speed4) : tmpSpeed.speed2;
      tmp_speeds->speed5[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed5
                                              + params.omega
                                              * (d_equ[5] - tmpSpeed.speed5) : tmpSpeed.speed7;
      tmp_speeds->speed6[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed6
                                              + params.omega
                                              * (d_equ[6] - tmpSpeed.speed6) : tmpSpeed.speed8;
      tmp_speeds->speed7[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed7
                                              + params.omega
                                              * (d_equ[7] - tmpSpeed.speed7) : tmpSpeed.speed5;
      tmp_speeds->speed8[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed.speed8
                                              + params.omega
                                              * (d_equ[8] - tmpSpeed.speed8) : tmpSpeed.speed6;
    }
  }
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, s_speeds* speeds, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 1;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (speeds->speed3[ii + jj*params.nx] - w1) > 0.f
        && (speeds->speed6[ii + jj*params.nx] - w2) > 0.f
        && (speeds->speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speeds->speed1[ii + jj*params.nx] += w1;
      speeds->speed5[ii + jj*params.nx] += w2;
      speeds->speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      speeds->speed3[ii + jj*params.nx] -= w1;
      speeds->speed6[ii + jj*params.nx] -= w2;
      speeds->speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float total_velocity(const t_param params, s_speeds* speeds, int* obstacles)
{
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked cells */
  for (int jj = 1; jj < params.ny + 1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += speeds->speed0[ii + jj*params.nx];
        local_density += speeds->speed1[ii + jj*params.nx];
        local_density += speeds->speed2[ii + jj*params.nx];
        local_density += speeds->speed3[ii + jj*params.nx];
        local_density += speeds->speed4[ii + jj*params.nx];
        local_density += speeds->speed5[ii + jj*params.nx];
        local_density += speeds->speed6[ii + jj*params.nx];
        local_density += speeds->speed7[ii + jj*params.nx];
        local_density += speeds->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (speeds->speed1[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed8[ii + jj*params.nx]
                      - (speeds->speed3[ii + jj*params.nx]
                         + speeds->speed6[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (speeds->speed2[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed6[ii + jj*params.nx]
                      - (speeds->speed4[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]
                         + speeds->speed8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
      }
    }
  }

  return tot_u;
}

float av_velocity(const t_param params, s_speeds* speeds, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        local_density += speeds->speed0[ii + jj*params.nx];
        local_density += speeds->speed1[ii + jj*params.nx];
        local_density += speeds->speed2[ii + jj*params.nx];
        local_density += speeds->speed3[ii + jj*params.nx];
        local_density += speeds->speed4[ii + jj*params.nx];
        local_density += speeds->speed5[ii + jj*params.nx];
        local_density += speeds->speed6[ii + jj*params.nx];
        local_density += speeds->speed7[ii + jj*params.nx];
        local_density += speeds->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (speeds->speed1[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed8[ii + jj*params.nx]
                      - (speeds->speed3[ii + jj*params.nx]
                         + speeds->speed6[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (speeds->speed2[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed6[ii + jj*params.nx]
                      - (speeds->speed4[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]
                         + speeds->speed8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, s_speeds *speeds, s_speeds *tmp_speeds,
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  speeds->speed0 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed1 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed2 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed3 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed4 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed5 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed6 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed7 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  speeds->speed8 = (float *) malloc(sizeof(float) * (params->ny * params->nx));

  if (speeds->speed0 == NULL) die("cannot allocate memory for speed0", __LINE__, __FILE__);
  if (speeds->speed1 == NULL) die("cannot allocate memory for speed1", __LINE__, __FILE__);
  if (speeds->speed2 == NULL) die("cannot allocate memory for speed2", __LINE__, __FILE__);
  if (speeds->speed3 == NULL) die("cannot allocate memory for speed3", __LINE__, __FILE__);
  if (speeds->speed4 == NULL) die("cannot allocate memory for speed4", __LINE__, __FILE__);
  if (speeds->speed5 == NULL) die("cannot allocate memory for speed5", __LINE__, __FILE__);
  if (speeds->speed6 == NULL) die("cannot allocate memory for speed6", __LINE__, __FILE__);
  if (speeds->speed7 == NULL) die("cannot allocate memory for speed7", __LINE__, __FILE__);
  if (speeds->speed8 == NULL) die("cannot allocate memory for speed8", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  tmp_speeds->speed0 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed1 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed2 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed3 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed4 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed5 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed6 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed7 = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  tmp_speeds->speed8 = (float *) malloc(sizeof(float) * (params->ny * params->nx));

  if (tmp_speeds->speed0 == NULL) die("cannot allocate memory for tmp_speed0", __LINE__, __FILE__);
  if (tmp_speeds->speed1 == NULL) die("cannot allocate memory for tmp_speed1", __LINE__, __FILE__);
  if (tmp_speeds->speed2 == NULL) die("cannot allocate memory for tmp_speed2", __LINE__, __FILE__);
  if (tmp_speeds->speed3 == NULL) die("cannot allocate memory for tmp_speed3", __LINE__, __FILE__);
  if (tmp_speeds->speed4 == NULL) die("cannot allocate memory for tmp_speed4", __LINE__, __FILE__);
  if (tmp_speeds->speed5 == NULL) die("cannot allocate memory for tmp_speed5", __LINE__, __FILE__);
  if (tmp_speeds->speed6 == NULL) die("cannot allocate memory for tmp_speed6", __LINE__, __FILE__);
  if (tmp_speeds->speed7 == NULL) die("cannot allocate memory for tmp_speed7", __LINE__, __FILE__);
  if (tmp_speeds->speed8 == NULL) die("cannot allocate memory for tmp_speed8", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      speeds->speed0[ii + jj*params->nx] = w0;
      /* axis directions */
      speeds->speed1[ii + jj*params->nx] = w1;
      speeds->speed2[ii + jj*params->nx] = w1;
      speeds->speed3[ii + jj*params->nx] = w1;
      speeds->speed4[ii + jj*params->nx] = w1;
      /* diagonals */
      speeds->speed5[ii + jj*params->nx] = w2;
      speeds->speed6[ii + jj*params->nx] = w2;
      speeds->speed7[ii + jj*params->nx] = w2;
      speeds->speed8[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, s_speeds* speeds, s_speeds* tmp_speeds,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(speeds->speed0);
  free(speeds->speed1);
  free(speeds->speed2);
  free(speeds->speed3);
  free(speeds->speed4);
  free(speeds->speed5);
  free(speeds->speed6);
  free(speeds->speed7);
  free(speeds->speed8);
  free(speeds);
  speeds = NULL;

  free(tmp_speeds->speed0);
  free(tmp_speeds->speed1);
  free(tmp_speeds->speed2);
  free(tmp_speeds->speed3);
  free(tmp_speeds->speed4);
  free(tmp_speeds->speed5);
  free(tmp_speeds->speed6);
  free(tmp_speeds->speed7);
  free(tmp_speeds->speed8);
  free(tmp_speeds);
  tmp_speeds = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, s_speeds* speeds, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, speeds, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, s_speeds* speeds)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += speeds->speed0[ii + jj*params.nx];
      total += speeds->speed1[ii + jj*params.nx];
      total += speeds->speed2[ii + jj*params.nx];
      total += speeds->speed3[ii + jj*params.nx];
      total += speeds->speed4[ii + jj*params.nx];
      total += speeds->speed5[ii + jj*params.nx];
      total += speeds->speed6[ii + jj*params.nx];
      total += speeds->speed7[ii + jj*params.nx];
      total += speeds->speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, s_speeds* speeds, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        local_density += speeds->speed0[ii + jj*params.nx];
        local_density += speeds->speed1[ii + jj*params.nx];
        local_density += speeds->speed2[ii + jj*params.nx];
        local_density += speeds->speed3[ii + jj*params.nx];
        local_density += speeds->speed4[ii + jj*params.nx];
        local_density += speeds->speed5[ii + jj*params.nx];
        local_density += speeds->speed6[ii + jj*params.nx];
        local_density += speeds->speed7[ii + jj*params.nx];
        local_density += speeds->speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (speeds->speed1[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed8[ii + jj*params.nx]
                      - (speeds->speed3[ii + jj*params.nx]
                         + speeds->speed6[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        float u_y = (speeds->speed2[ii + jj*params.nx]
                      + speeds->speed5[ii + jj*params.nx]
                      + speeds->speed6[ii + jj*params.nx]
                      - (speeds->speed4[ii + jj*params.nx]
                         + speeds->speed7[ii + jj*params.nx]
                         + speeds->speed8[ii + jj*params.nx]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
