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

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile, t_param* params, float** speed0_ptr, float** speed1_ptr, float** speed2_ptr, float** speed3_ptr, float** speed4_ptr, float** speed5_ptr, float** speed6_ptr, float** speed7_ptr, float** speed8_ptr, float** tmp_speed0_ptr, float** tmp_speed1_ptr, float** tmp_speed2_ptr, float** tmp_speed3_ptr, float** tmp_speed4_ptr, float** tmp_speed5_ptr, float** tmp_speed6_ptr, float** tmp_speed7_ptr, float** tmp_speed8_ptr, int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
*/
float timestep(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int* obstacles);
int accelerate_flow(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);
int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8);

/* compute average velocity */
float av_velocity(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles);

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
  float* speed0 = NULL;
  float* speed1 = NULL;
  float* speed2 = NULL;
  float* speed3 = NULL;
  float* speed4 = NULL;
  float* speed5 = NULL;
  float* speed6 = NULL;
  float* speed7 = NULL;
  float* speed8 = NULL;
  float* tmp_speed0 = NULL;
  float* tmp_speed1 = NULL;
  float* tmp_speed2 = NULL;
  float* tmp_speed3 = NULL;
  float* tmp_speed4 = NULL;
  float* tmp_speed5 = NULL;
  float* tmp_speed6 = NULL;
  float* tmp_speed7 = NULL;
  float* tmp_speed8 = NULL;
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
  int *send_cnts = (int *) malloc(sizeof(int) * worldSize);
  int *row_cnts = (int *) malloc(sizeof(int) * worldSize);
  int *displs = (int *) malloc(sizeof(int) * worldSize);

  /* initialise our data structures and load values from file */
  if (rank == 0) {
    initialise(paramfile, obstaclefile, &params, &speed0, &speed1, &speed2, &speed3, &speed4, &speed5, &speed6, &speed7, &speed8, &tmp_speed0, &tmp_speed1, &tmp_speed2, &tmp_speed3, &tmp_speed4, &tmp_speed5, &tmp_speed6, &tmp_speed7, &tmp_speed8, &obstacles, &av_vels);

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

  int N = sub_params.nx * (sub_params.ny + 2);

  int *sub_obstacles = (int *) malloc(sizeof(int) * N);

  float* sub_speed0 = (float *) malloc(sizeof(float) * N);
  float* sub_speed1 = (float *) malloc(sizeof(float) * N);
  float* sub_speed2 = (float *) malloc(sizeof(float) * N);
  float* sub_speed3 = (float *) malloc(sizeof(float) * N);
  float* sub_speed4 = (float *) malloc(sizeof(float) * N);
  float* sub_speed5 = (float *) malloc(sizeof(float) * N);
  float* sub_speed6 = (float *) malloc(sizeof(float) * N);
  float* sub_speed7 = (float *) malloc(sizeof(float) * N);
  float* sub_speed8 = (float *) malloc(sizeof(float) * N);

  float* sub_tmp_speed0 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed1 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed2 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed3 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed4 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed5 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed6 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed7 = (float *) malloc(sizeof(float) * N);
  float* sub_tmp_speed8 = (float *) malloc(sizeof(float) * N);

  MPI_Scatterv(obstacles, send_cnts, displs, MPI_INT, (sub_obstacles + sub_params.nx), send_cnts[rank], MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed0, send_cnts, displs, MPI_FLOAT, (sub_speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed1, send_cnts, displs, MPI_FLOAT, (sub_speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed2, send_cnts, displs, MPI_FLOAT, (sub_speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed3, send_cnts, displs, MPI_FLOAT, (sub_speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed4, send_cnts, displs, MPI_FLOAT, (sub_speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed5, send_cnts, displs, MPI_FLOAT, (sub_speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed6, send_cnts, displs, MPI_FLOAT, (sub_speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed7, send_cnts, displs, MPI_FLOAT, (sub_speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(speed8, send_cnts, displs, MPI_FLOAT, (sub_speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Request haloRequests[NSPEEDS];
  MPI_Status haloStatuses[NSPEEDS];

  float* sendbuf0 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf1 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf2 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf3 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf4 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf5 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf6 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf7 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* sendbuf8 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);

  float* recvbuf0 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf1 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf2 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf3 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf4 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf5 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf6 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf7 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);
  float* recvbuf8 = (float *) malloc(sizeof(float) * 2 * cols_per_proc);

  float* pSub_speed0 = sub_speed0;
  float* pSub_speed1 = sub_speed1;
  float* pSub_speed2 = sub_speed2;
  float* pSub_speed3 = sub_speed3;
  float* pSub_speed4 = sub_speed4;
  float* pSub_speed5 = sub_speed5;
  float* pSub_speed6 = sub_speed6;
  float* pSub_speed7 = sub_speed7;
  float* pSub_speed8 = sub_speed8;
  float* pSub_tmp_speed0 = sub_tmp_speed0;
  float* pSub_tmp_speed1 = sub_tmp_speed1;
  float* pSub_tmp_speed2 = sub_tmp_speed2;
  float* pSub_tmp_speed3 = sub_tmp_speed3;
  float* pSub_tmp_speed4 = sub_tmp_speed4;
  float* pSub_tmp_speed5 = sub_tmp_speed5;
  float* pSub_tmp_speed6 = sub_tmp_speed6;
  float* pSub_tmp_speed7 = sub_tmp_speed7;
  float* pSub_tmp_speed8 = sub_tmp_speed8;
  float local_tot_vel, global_tot_vel;

  if (rank == 0) {
    gettimeofday(&timstr, NULL);
    tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  }


  /* iterate for maxIters timesteps */
  #pragma omp target enter data map(to: sub_speed0[0:N], sub_speed1[0:N], sub_speed2[0:N], sub_speed3[0:N], sub_speed4[0:N], sub_speed5[0:N], sub_speed6[0:N], sub_speed7[0:N], sub_speed8[0:N], sub_tmp_speed0[0:N], sub_tmp_speed1[0:N], sub_tmp_speed2[0:N], sub_tmp_speed3[0:N], sub_tmp_speed4[0:N], sub_tmp_speed5[0:N], sub_tmp_speed6[0:N], sub_tmp_speed7[0:N], sub_tmp_speed8[0:N])
  for (int tt = 0; tt < params.maxIters; tt++)
  {
    if ((worldSize - 1) == rank) {
      accelerate_flow(sub_params, pSub_speed0, pSub_speed1, pSub_speed2, pSub_speed3, pSub_speed4, pSub_speed5, pSub_speed6, pSub_speed7, pSub_speed8, sub_obstacles);
    }

    memcpy(sendbuf0, pSub_speed0 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf0 + sub_params.nx, pSub_speed0 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf1, pSub_speed1 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf1 + sub_params.nx, pSub_speed1 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf2, pSub_speed2 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf2 + sub_params.nx, pSub_speed2 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf3, pSub_speed3 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf3 + sub_params.nx, pSub_speed3 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf4, pSub_speed4 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf4 + sub_params.nx, pSub_speed4 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf5, pSub_speed5 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf5 + sub_params.nx, pSub_speed5 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf6, pSub_speed6 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf6 + sub_params.nx, pSub_speed6 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf7, pSub_speed7 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf7 + sub_params.nx, pSub_speed7 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);
    memcpy(sendbuf8, pSub_speed8 + sub_params.nx, sizeof(float) * sub_params.nx);
    memcpy(sendbuf8 + sub_params.nx, pSub_speed8 + (sub_params.ny * sub_params.nx), sizeof(float) * sub_params.nx);

    MPI_Ineighbor_alltoall(sendbuf0, sub_params.nx, MPI_FLOAT, recvbuf0, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[0]);
    MPI_Ineighbor_alltoall(sendbuf1, sub_params.nx, MPI_FLOAT, recvbuf1, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[1]);
    MPI_Ineighbor_alltoall(sendbuf2, sub_params.nx, MPI_FLOAT, recvbuf2, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[2]);
    MPI_Ineighbor_alltoall(sendbuf3, sub_params.nx, MPI_FLOAT, recvbuf3, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[3]);
    MPI_Ineighbor_alltoall(sendbuf4, sub_params.nx, MPI_FLOAT, recvbuf4, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[4]);
    MPI_Ineighbor_alltoall(sendbuf5, sub_params.nx, MPI_FLOAT, recvbuf5, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[5]);
    MPI_Ineighbor_alltoall(sendbuf6, sub_params.nx, MPI_FLOAT, recvbuf6, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[6]);
    MPI_Ineighbor_alltoall(sendbuf7, sub_params.nx, MPI_FLOAT, recvbuf7, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[7]);
    MPI_Ineighbor_alltoall(sendbuf8, sub_params.nx, MPI_FLOAT, recvbuf8, sub_params.nx, MPI_FLOAT, cart_world, &haloRequests[8]);

    MPI_Waitall(NSPEEDS, haloRequests, haloStatuses);

    if (worldSize != 2) {
      memcpy(pSub_speed0 + ((sub_params.ny + 1) * sub_params.nx), recvbuf0 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed0, recvbuf0, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed1 + ((sub_params.ny + 1) * sub_params.nx), recvbuf1 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed1, recvbuf1, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed2 + ((sub_params.ny + 1) * sub_params.nx), recvbuf2 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed2, recvbuf2, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed3 + ((sub_params.ny + 1) * sub_params.nx), recvbuf3 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed3, recvbuf3, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed4, recvbuf4, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed5 + ((sub_params.ny + 1) * sub_params.nx), recvbuf5 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed5, recvbuf5, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed6 + ((sub_params.ny + 1) * sub_params.nx), recvbuf6 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed6, recvbuf6, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed7, recvbuf7, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed8, recvbuf8, sizeof(float) * sub_params.nx);
    } else {
      memcpy(pSub_speed0 + ((sub_params.ny + 1) * sub_params.nx), recvbuf0, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed0, recvbuf0 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed1 + ((sub_params.ny + 1) * sub_params.nx), recvbuf1, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed1, recvbuf1 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed2 + ((sub_params.ny + 1) * sub_params.nx), recvbuf2, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed2, recvbuf2 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed3 + ((sub_params.ny + 1) * sub_params.nx), recvbuf3, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed3, recvbuf3 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed4 + ((sub_params.ny + 1) * sub_params.nx), recvbuf4, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed4, recvbuf4 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed5 + ((sub_params.ny + 1) * sub_params.nx), recvbuf5, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed5, recvbuf5 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed6 + ((sub_params.ny + 1) * sub_params.nx), recvbuf6, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed6, recvbuf6 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed7 + ((sub_params.ny + 1) * sub_params.nx), recvbuf7, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed7, recvbuf7 + sub_params.nx, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed8 + ((sub_params.ny + 1) * sub_params.nx), recvbuf8, sizeof(float) * sub_params.nx);
      memcpy(pSub_speed8, recvbuf8 + sub_params.nx, sizeof(float) * sub_params.nx);
    }

    local_tot_vel = timestep(sub_params, pSub_speed0, pSub_speed1, pSub_speed2, pSub_speed3, pSub_speed4, pSub_speed5, pSub_speed6, pSub_speed7, pSub_speed8, pSub_tmp_speed0, pSub_tmp_speed1, pSub_tmp_speed2, pSub_tmp_speed3, pSub_tmp_speed4, pSub_tmp_speed5, pSub_tmp_speed6, pSub_tmp_speed7, pSub_tmp_speed8, sub_obstacles);
    pSub_speed0 = (tt % 2) ? sub_speed0 : sub_tmp_speed0;
    pSub_speed1 = (tt % 2) ? sub_speed1 : sub_tmp_speed1;
    pSub_speed2 = (tt % 2) ? sub_speed2 : sub_tmp_speed2;
    pSub_speed3 = (tt % 2) ? sub_speed3 : sub_tmp_speed3;
    pSub_speed4 = (tt % 2) ? sub_speed4 : sub_tmp_speed4;
    pSub_speed5 = (tt % 2) ? sub_speed5 : sub_tmp_speed5;
    pSub_speed6 = (tt % 2) ? sub_speed6 : sub_tmp_speed6;
    pSub_speed7 = (tt % 2) ? sub_speed7 : sub_tmp_speed7;
    pSub_speed8 = (tt % 2) ? sub_speed8 : sub_tmp_speed8;
    pSub_tmp_speed0 = (tt % 2) ? sub_tmp_speed0 : sub_speed0;
    pSub_tmp_speed1 = (tt % 2) ? sub_tmp_speed1 : sub_speed1;
    pSub_tmp_speed2 = (tt % 2) ? sub_tmp_speed2 : sub_speed2;
    pSub_tmp_speed3 = (tt % 2) ? sub_tmp_speed3 : sub_speed3;
    pSub_tmp_speed4 = (tt % 2) ? sub_tmp_speed4 : sub_speed4;
    pSub_tmp_speed5 = (tt % 2) ? sub_tmp_speed5 : sub_speed5;
    pSub_tmp_speed6 = (tt % 2) ? sub_tmp_speed6 : sub_speed6;
    pSub_tmp_speed7 = (tt % 2) ? sub_tmp_speed7 : sub_speed7;
    pSub_tmp_speed8 = (tt % 2) ? sub_tmp_speed8 : sub_speed8;

    MPI_Reduce(&local_tot_vel, &global_tot_vel, 1, MPI_FLOAT, MPI_SUM, 0, cart_world);
    if (rank == 0) {
      av_vels[tt] = global_tot_vel / (float) tot_cells;
    }
#ifdef DEBUG
  if (rank == 0) {
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(sub_params, pSub_speed0, pSub_speed1, pSub_speed2, pSub_speed3, pSub_speed4, pSub_speed5, pSub_speed6, pSub_speed7, pSub_speed8));
  }
#endif
  }
  #pragma omp target exit data map(from: sub_speed0[0:N], sub_speed1[0:N], sub_speed2[0:N], sub_speed3[0:N], sub_speed4[0:N], sub_speed5[0:N], sub_speed6[0:N], sub_speed7[0:N], sub_speed8[0:N], sub_tmp_speed0[0:N], sub_tmp_speed1[0:N], sub_tmp_speed2[0:N], sub_tmp_speed3[0:N], sub_tmp_speed4[0:N], sub_tmp_speed5[0:N], sub_tmp_speed6[0:N], sub_tmp_speed7[0:N], sub_tmp_speed8[0:N])

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

  MPI_Gatherv((sub_speed0 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed0, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed1 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed1, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed2 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed2, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed3 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed3, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed4 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed4, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed5 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed5, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed6 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed6, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed7 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed7, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gatherv((sub_speed8 + sub_params.nx), send_cnts[rank], MPI_FLOAT, speed8, send_cnts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

  MPI_Finalize();

  if (rank == 0) {
    /* write final values and free memory */
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles, av_vels);
    finalise(&params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, tmp_speed0, tmp_speed1, tmp_speed2, tmp_speed3, tmp_speed4, tmp_speed5, tmp_speed6, tmp_speed7, tmp_speed8, &obstacles, &av_vels);
  }

  return EXIT_SUCCESS;
}

float timestep(const t_param params, float* __restrict__ speed0, float* __restrict__ speed1, float* __restrict__ speed2, float* __restrict__ speed3, float* __restrict__ speed4, float* __restrict__ speed5, float* __restrict__ speed6, float* __restrict__ speed7, float* __restrict__ speed8, float* __restrict__ tmp_speed0, float* __restrict__ tmp_speed1, float* __restrict__ tmp_speed2, float* __restrict__ tmp_speed3, float* __restrict__ tmp_speed4, float* __restrict__ tmp_speed5, float* __restrict__ tmp_speed6, float* __restrict__ tmp_speed7, float* __restrict__ tmp_speed8, int* __restrict__ obstacles) {
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8;
  float tot_u = 0;

  int y_n, x_e, y_s, x_w;
  float local_density, u_x, u_y, u_sq;
  /* loop over _all_ cells */
  #pragma omp target teams distribute parallel for simd map(tofrom:tot_u) private(tmpSpeed0, tmpSpeed1, tmpSpeed2, tmpSpeed3, tmpSpeed4, tmpSpeed5, tmpSpeed6, tmpSpeed7, tmpSpeed8, y_n, x_e, y_s, x_w, local_density, u_x, u_y, u_sq) reduction(+:tot_u)
  for (int jj = 1; jj < params.ny + 1; jj++) {
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
      tmpSpeed0 = speed0[ii + jj*params.nx]; /* central cell, no movement */
      tmpSpeed1 = speed1[x_w + jj*params.nx]; /* east */
      tmpSpeed2 = speed2[ii + y_s*params.nx]; /* north */
      tmpSpeed3 = speed3[x_e + jj*params.nx]; /* west */
      tmpSpeed4 = speed4[ii + y_n*params.nx]; /* south */
      tmpSpeed5 = speed5[x_w + y_s*params.nx]; /* north-east */
      tmpSpeed6 = speed6[x_e + y_s*params.nx]; /* north-west */
      tmpSpeed7 = speed7[x_e + y_n*params.nx]; /* south-west */
      tmpSpeed8 = speed8[x_w + y_n*params.nx]; /* south-east */

      /* compute local density total */
      local_density = 0.f;

      local_density += tmpSpeed0;
      local_density += tmpSpeed1;
      local_density += tmpSpeed2;
      local_density += tmpSpeed3;
      local_density += tmpSpeed4;
      local_density += tmpSpeed5;
      local_density += tmpSpeed6;
      local_density += tmpSpeed7;
      local_density += tmpSpeed8;

      /* compute x velocity component */
      u_x = (tmpSpeed1
                    + tmpSpeed5
                    + tmpSpeed8
                    - (tmpSpeed3
                        + tmpSpeed6
                        + tmpSpeed7))
                    / local_density;
      /* compute y velocity component */
      u_y = (tmpSpeed2
                    + tmpSpeed5
                    + tmpSpeed6
                    - (tmpSpeed4
                        + tmpSpeed7
                        + tmpSpeed8))
                    / local_density;

      /* velocity squared */
      u_sq = u_x * u_x + u_y * u_y;
      tot_u += !obstacles[ii + jj*params.nx] ? sqrtf(u_sq) : 0;

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
      tmp_speed0[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed0
                                                + params.omega
                                              * (d_equ[0] - tmpSpeed0) : tmpSpeed0;
      tmp_speed1[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed1
                                              + params.omega
                                              * (d_equ[1] - tmpSpeed1) : tmpSpeed3;
      tmp_speed2[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed2
                                              + params.omega
                                              * (d_equ[2] - tmpSpeed2) : tmpSpeed4;
      tmp_speed3[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed3
                                              + params.omega
                                              * (d_equ[3] - tmpSpeed3) : tmpSpeed1;
      tmp_speed4[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed4
                                              + params.omega
                                              * (d_equ[4] - tmpSpeed4) : tmpSpeed2;
      tmp_speed5[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed5
                                              + params.omega
                                              * (d_equ[5] - tmpSpeed5) : tmpSpeed7;
      tmp_speed6[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed6
                                              + params.omega
                                              * (d_equ[6] - tmpSpeed6) : tmpSpeed8;
      tmp_speed7[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed7
                                              + params.omega
                                              * (d_equ[7] - tmpSpeed7) : tmpSpeed5;
      tmp_speed8[ii + jj*params.nx] = !obstacles[ii + jj*params.nx] ? tmpSpeed8
                                              + params.omega
                                              * (d_equ[8] - tmpSpeed8) : tmpSpeed6;
    }
  }
  return tot_u;
}

int accelerate_flow(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 1;

  #pragma omp target teams distribute parallel for simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (speed3[ii + jj*params.nx] - w1) > 0.f
        && (speed6[ii + jj*params.nx] - w2) > 0.f
        && (speed7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speed1[ii + jj*params.nx] += w1;
      speed5[ii + jj*params.nx] += w2;
      speed8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      speed3[ii + jj*params.nx] -= w1;
      speed6[ii + jj*params.nx] -= w2;
      speed7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
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

        local_density += speed0[ii + jj*params.nx];
        local_density += speed1[ii + jj*params.nx];
        local_density += speed2[ii + jj*params.nx];
        local_density += speed3[ii + jj*params.nx];
        local_density += speed4[ii + jj*params.nx];
        local_density += speed5[ii + jj*params.nx];
        local_density += speed6[ii + jj*params.nx];
        local_density += speed7[ii + jj*params.nx];
        local_density += speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (speed1[ii + jj*params.nx]
                      + speed5[ii + jj*params.nx]
                      + speed8[ii + jj*params.nx]
                      - (speed3[ii + jj*params.nx]
                         + speed6[ii + jj*params.nx]
                         + speed7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (speed2[ii + jj*params.nx]
                      + speed5[ii + jj*params.nx]
                      + speed6[ii + jj*params.nx]
                      - (speed4[ii + jj*params.nx]
                         + speed7[ii + jj*params.nx]
                         + speed8[ii + jj*params.nx]))
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
               t_param* params, float** speed0_ptr, float** speed1_ptr, float** speed2_ptr, float** speed3_ptr, float** speed4_ptr, float** speed5_ptr, float** speed6_ptr, float** speed7_ptr, float** speed8_ptr, float** tmp_speed0_ptr, float** tmp_speed1_ptr, float** tmp_speed2_ptr, float** tmp_speed3_ptr, float** tmp_speed4_ptr, float** tmp_speed5_ptr, float** tmp_speed6_ptr, float** tmp_speed7_ptr, float** tmp_speed8_ptr, int** obstacles_ptr, float** av_vels_ptr)
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
  *speed0_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed1_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed2_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed3_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed4_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed5_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed6_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed7_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *speed8_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));

  if (*speed0_ptr == NULL) die("cannot allocate memory for speed0", __LINE__, __FILE__);
  if (*speed1_ptr == NULL) die("cannot allocate memory for speed1", __LINE__, __FILE__);
  if (*speed2_ptr == NULL) die("cannot allocate memory for speed2", __LINE__, __FILE__);
  if (*speed3_ptr == NULL) die("cannot allocate memory for speed3", __LINE__, __FILE__);
  if (*speed4_ptr == NULL) die("cannot allocate memory for speed4", __LINE__, __FILE__);
  if (*speed5_ptr == NULL) die("cannot allocate memory for speed5", __LINE__, __FILE__);
  if (*speed6_ptr == NULL) die("cannot allocate memory for speed6", __LINE__, __FILE__);
  if (*speed7_ptr == NULL) die("cannot allocate memory for speed7", __LINE__, __FILE__);
  if (*speed8_ptr == NULL) die("cannot allocate memory for speed8", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_speed0_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed1_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed2_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed3_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed4_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed5_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed6_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed7_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));
  *tmp_speed8_ptr = (float *) malloc(sizeof(float) * (params->ny * params->nx));

  if (*tmp_speed0_ptr == NULL) die("cannot allocate memory for tmp_speed0", __LINE__, __FILE__);
  if (*tmp_speed1_ptr == NULL) die("cannot allocate memory for tmp_speed1", __LINE__, __FILE__);
  if (*tmp_speed2_ptr == NULL) die("cannot allocate memory for tmp_speed2", __LINE__, __FILE__);
  if (*tmp_speed3_ptr == NULL) die("cannot allocate memory for tmp_speed3", __LINE__, __FILE__);
  if (*tmp_speed4_ptr == NULL) die("cannot allocate memory for tmp_speed4", __LINE__, __FILE__);
  if (*tmp_speed5_ptr == NULL) die("cannot allocate memory for tmp_speed5", __LINE__, __FILE__);
  if (*tmp_speed6_ptr == NULL) die("cannot allocate memory for tmp_speed6", __LINE__, __FILE__);
  if (*tmp_speed7_ptr == NULL) die("cannot allocate memory for tmp_speed7", __LINE__, __FILE__);
  if (*tmp_speed8_ptr == NULL) die("cannot allocate memory for tmp_speed8", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (int *) malloc(sizeof(int) * (params->ny * params->nx));

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
      (*speed0_ptr)[ii + jj*params->nx] = w0;
      /* axis directions */
      (*speed1_ptr)[ii + jj*params->nx] = w1;
      (*speed2_ptr)[ii + jj*params->nx] = w1;
      (*speed3_ptr)[ii + jj*params->nx] = w1;
      (*speed4_ptr)[ii + jj*params->nx] = w1;
      /* diagonals */
      (*speed5_ptr)[ii + jj*params->nx] = w2;
      (*speed6_ptr)[ii + jj*params->nx] = w2;
      (*speed7_ptr)[ii + jj*params->nx] = w2;
      (*speed8_ptr)[ii + jj*params->nx] = w2;
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

int finalise(const t_param* params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, float* tmp_speed0, float* tmp_speed1, float* tmp_speed2, float* tmp_speed3, float* tmp_speed4, float* tmp_speed5, float* tmp_speed6, float* tmp_speed7, float* tmp_speed8, int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(speed0);
  free(speed1);
  free(speed2);
  free(speed3);
  free(speed4);
  free(speed5);
  free(speed6);
  free(speed7);
  free(speed8);

  free(tmp_speed0);
  free(tmp_speed1);
  free(tmp_speed2);
  free(tmp_speed3);
  free(tmp_speed4);
  free(tmp_speed5);
  free(tmp_speed6);
  free(tmp_speed7);
  free(tmp_speed8);

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, speed0, speed1, speed2, speed3, speed4, speed5, speed6, speed7, speed8, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += speed0[ii + jj*params.nx];
      total += speed1[ii + jj*params.nx];
      total += speed2[ii + jj*params.nx];
      total += speed3[ii + jj*params.nx];
      total += speed4[ii + jj*params.nx];
      total += speed5[ii + jj*params.nx];
      total += speed6[ii + jj*params.nx];
      total += speed7[ii + jj*params.nx];
      total += speed8[ii + jj*params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, float* speed0, float* speed1, float* speed2, float* speed3, float* speed4, float* speed5, float* speed6, float* speed7, float* speed8, int* obstacles, float* av_vels)
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

        local_density += speed0[ii + jj*params.nx];
        local_density += speed1[ii + jj*params.nx];
        local_density += speed2[ii + jj*params.nx];
        local_density += speed3[ii + jj*params.nx];
        local_density += speed4[ii + jj*params.nx];
        local_density += speed5[ii + jj*params.nx];
        local_density += speed6[ii + jj*params.nx];
        local_density += speed7[ii + jj*params.nx];
        local_density += speed8[ii + jj*params.nx];

        /* compute x velocity component */
        float u_x = (speed1[ii + jj*params.nx]
                      + speed5[ii + jj*params.nx]
                      + speed8[ii + jj*params.nx]
                      - (speed3[ii + jj*params.nx]
                         + speed6[ii + jj*params.nx]
                         + speed7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        float u_y = (speed2[ii + jj*params.nx]
                      + speed5[ii + jj*params.nx]
                      + speed6[ii + jj*params.nx]
                      - (speed4[ii + jj*params.nx]
                         + speed7[ii + jj*params.nx]
                         + speed8[ii + jj*params.nx]))
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
