/**
A New Parallel Sorting Algorithm based on Odd-Even Mergesort
http://ieeexplore.ieee.org/document/4135254/
**/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

int comparefunc(const void *a, const void *b){
	if(*(float*)a < *(float*)b) return -1;
	if(*(float*)a > *(float*)b) return 1;
	return 0;
}

int isSwap;
void swap_float_ptr(float **a, float **b){
	float *t;
	t = *a;
	*a = *b;
	*b = t;
}
void swap_float(float *a, float *b){
	float t = *a;
	*a = *b;
	*b = t;
	isSwap = 1;
}
void print(float *buf, int n, int rank, int size, char t){
	int i;
	for(i=0;i<n;i++){
		printf("%cproc #%02d/%02d, buf[%d]=%f\n", t, rank, size, i, buf[i]);
	}
	puts("");
}

int main(int argc, char** argv){
	assert(argc == 4); /* [exe] #floats in_file out_file */
	MPI_Init(&argc, &argv);
	double CPUtime=0,COMtime=0,IOtime=0,t1,t2,t_time;
	t1 = MPI_Wtime();
	
	int N = atoi(argv[1]);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int n, qot, rmd; /* #floats(cur_proc) Quotient remainder*/
	int n_pre,n_nxt; // to record the #floats of adjacent procs.
	qot = N/size;
	rmd = N%size;
	n = qot + (rank<rmd);
	int ofs = rank*qot + ((rank<rmd) ? rank:rmd);
	if(!rank) n_pre = 0;
	else n_pre = qot + (rank-1<rmd);
	if(rank == size-1) n_nxt = 0;
	else n_nxt = qot + (rank+1<rmd);
	float *buf1 = (float*)malloc(n*sizeof(float));
	// the adjacent buf may contain one more number.
	float *buf2 = (float*)malloc((n+1)*sizeof(float));
	float *buf3 = (float*)malloc(n*sizeof(float));
	
	t2 = MPI_Wtime();
	CPUtime += t2-t1;
	
	t1 = MPI_Wtime();
	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at(fh, ofs*sizeof(MPI_FLOAT), (void*)buf1, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	t2 = MPI_Wtime();
	IOtime += t2-t1;
	
	t1 = MPI_Wtime();
	// local Quick sort
	qsort(buf1, (size_t)n, sizeof(float), comparefunc);
	
	int i,j,k,cnt,cycNum = (size+1)>>1;
	//MPI_Request isreq, irreq; // may be used
	float t;
	int ti, unsorted = 1;
	t2 = MPI_Wtime();
	CPUtime += t2-t1;
	
	cnt = 0;
	while(unsorted){
	//for(cnt=0;cnt<cycNum;cnt++){
		//printf("cyc %d @%d/%d\n",cnt,rank,size);
		isSwap = 0;
		/*** Even phase: 0&1, 2&3, ... ***/
		if(rank&1){ /* send to & receive from the left */
			if(n && n_pre){
				t1 = MPI_Wtime();
				//MPI_Isend(buf1, n, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD, &isreq);
				MPI_Send(buf1, n, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);
				MPI_Recv(buf2, n_pre, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//MPI_Wait(&isreq, MPI_STATUS_IGNORE);
				t2 = MPI_Wtime();
				COMtime += t2-t1;
				
				// start to merge
				t1 = MPI_Wtime();
				i = n-1; j = n_pre-1; k = n-1;
				while(k>=0&&i>=0&&j>=0){
					//printf("k=%d,n=%d,i=%d,j=%d\n",k,n,i,j);
					if(buf1[i] >= buf2[j]){
						buf3[k--] = buf1[i--];
						//if(rank==1) printf("buf3[%d]=%f copy from %f among(%f,%f)\n",k,buf3[k],buf1[i],buf1[i],buf2[j]);
					}else{
						buf3[k--] = buf2[j--];
						//if(rank==1)printf("buf3[%d]=%f copy from %f among(%f,%f)\n",k,buf3[k],buf2[j],buf1[i],buf2[j]);
						isSwap = 1;
						//printf("Swap between (%d, %d) / %d\n",rank,rank-1,size);
					}
				}// it is not possible to have more floats than (rank-1), so no need to i>=0 or j>=0
				assert(k==-1);
				swap_float_ptr(&buf1,&buf3);
				t2 = MPI_Wtime();
				CPUtime += t2-t1;
			}
		}
		else{ // send to or receive from the right
			if(n && n_nxt){
				t1 = MPI_Wtime();
				//MPI_Isend(buf1, n, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD, &isreq);
				MPI_Recv(buf2, n_nxt, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(buf1, n, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
				//print(buf2,n_nxt,rank+1,size,rank+'0');
				//MPI_Wait(&isreq, MPI_STATUS_IGNORE);
				t2 = MPI_Wtime();
				COMtime += t2-t1;
				
				// start to merge
				t1 = MPI_Wtime();
				i = j = k = 0;
				while(k<n&&i<n&&j<n_nxt){
					if(buf1[i] <= buf2[j]){
						buf3[k++] = buf1[i++];
					}else{
						buf3[k++] = buf2[j++];
						isSwap = 1;
						//printf("Swap between (%d, %d) / %d\n",rank,rank+1,size);
					}
				}
				while(k<n){ // n_nxt may be smaller than n
					//printf("k/n=%d/%d,i=%d,j=%d,k=%d @proc%d\n",k,n,i,j,k,rank);
					buf3[k++] = buf1[i++];
				}
				assert(k==n);
				swap_float_ptr(&buf1,&buf3);
				t2 = MPI_Wtime();
				CPUtime += t2-t1;
			}
		}
		
		/*** Odd phase: 1&2, 3&4, ... ***/
		if(rank&1){ /* send to or receive from the right */
			if(n && n_nxt){
				t1 = MPI_Wtime();
				//MPI_Isend(buf1, n, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD, &isreq);
				MPI_Send(buf1, n, MPI_FLOAT, rank+1, (rank), MPI_COMM_WORLD);
				MPI_Recv(buf2, n_nxt, MPI_FLOAT, rank+1, (rank+1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//MPI_Wait(&isreq, MPI_STATUS_IGNORE);
				t2 = MPI_Wtime();
				COMtime += t2-t1;
				
				// start to merge
				t1 = MPI_Wtime();
				i = j = k = 0;
				while(k<n&&i<n&&j<n_nxt){
					if(buf1[i] <= buf2[j]){
						buf3[k++] = buf1[i++];
					}else{
						buf3[k++] = buf2[j++];
						isSwap = 1;
						//printf("Swap between (%d, %d) / %d\n",rank,rank+1,size);
					}
				}
				while(k<n){ // n_nxt may be smaller than n
					//printf("k/n=%d/%d,i=%d,j=%d,k=%d @proc%d\n",k,n,i,j,k,rank);
					buf3[k++] = buf1[i++];
				}
				assert(k==n);
				swap_float_ptr(&buf1,&buf3);
				t2 = MPI_Wtime();
				CPUtime += t2-t1;
			}
		}else{ // send to or receive from  the left
			if(n && n_pre){
				t1 = MPI_Wtime();
				//MPI_Isend(buf1, n, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD, &isreq);
				MPI_Recv(buf2, n_pre, MPI_FLOAT, rank-1, (rank-1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(buf1, n, MPI_FLOAT, rank-1, (rank), MPI_COMM_WORLD);
				//MPI_Wait(&isreq, MPI_STATUS_IGNORE);
				t2 = MPI_Wtime();
				COMtime += t2-t1;
				
				// start to merge
				t1 = MPI_Wtime();
				i = n-1; j = n_pre-1; k = n-1;
				while(k>=0&&i>=0&&j>=0){
					if(buf1[i] >= buf2[j]){
						buf3[k--] = buf1[i--];
					}else{
						buf3[k--] = buf2[j--];
						isSwap = 1;
						//printf("Swap between (%d, %d) / %d\n",rank,rank-1,size);
					}
				}// it is not possible to have more floats than (rank-1), so no need to i>=0 or j>=0
				assert(k==-1);
				swap_float_ptr(&buf1,&buf3);
				t2 = MPI_Wtime();
				CPUtime += t2-t1;
			}
		}
		// all reduce (isSwap);
		t1 = MPI_Wtime();
		MPI_Allreduce(&isSwap, &unsorted, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		COMtime += t2-t1;
		//printf("cycle %d,isSwap=%d unsorted=%d @proc %2d\n",cnt,isSwap,unsorted,rank);
		//++cnt;
		
	}
	//printf("cnt=%d @%d/%d\n",cnt,rank,size);
	
	t1 = MPI_Wtime();
	// mpi_write_at ... 
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	MPI_File_write_at(fh, ofs*sizeof(MPI_FLOAT), buf1, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	t2 = MPI_Wtime();
	IOtime += (t2-t1);
	
	t1 = MPI_Wtime();
	free(buf1);
	free(buf2);
	free(buf3);
	t2 = MPI_Wtime();
	CPUtime += t2-t1;
	
	t_time = IOtime;
	MPI_Reduce(&t_time, &IOtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	t_time = COMtime;
	MPI_Reduce(&t_time, &COMtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	t_time = CPUtime;
	MPI_Reduce(&t_time, &CPUtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	
	if(!rank){
		printf("Total I/O time: %lf\n", IOtime);
		printf("Total Communication time: %lf\n", COMtime);
		printf("Total CPU time: %lf\n", CPUtime);
	}
	MPI_Finalize();
	return 0;
}
