#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>



int isSwap;
void swap_float(float *a, float *b){
	float t = *a;
	*a = *b;
	*b = t;
	isSwap = 1;
}
void print(float *buf, int n, int rank, int size){
	int i;
	for(i=0;i<n;i++){
		printf("proc #%02d/%02d, buf[%d]=%f\n", rank, size, i, buf[i]);
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
	qot = N/size;
	rmd = N%size;
	n = qot + (rank<rmd);
	int ofs = rank*qot + ((rank<rmd) ? rank:rmd);
	
	float *buf = (float*)malloc(n*sizeof(float));
	
	t2 = MPI_Wtime();
	CPUtime += t2-t1;
	
	t1 = MPI_Wtime();
	MPI_File fh;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	MPI_File_read_at(fh, ofs*sizeof(MPI_FLOAT), (void*)buf, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	t2 = MPI_Wtime();
	IOtime = t2-t1;
	
	int i,fx,fy;
	float t;
	int key, ti, unsorted = 1,cnt = 0;
	while(unsorted){
		isSwap = 0;
		
		t1 = MPI_Wtime();
		/*** even_phase(rank,buf,ofs); ***/
		// if my 'zero' is odd => send
		// if my 'end' is even => receive & return
		fx = rank && (ofs&1) && (n>0); // is 'zero' odd idx ?
		fy = (rank<size-1) && ((ofs+n)&1) && (ofs+n<N); // is 'end' even idx? (end+1 would be odd)

		if(ofs&1) i = 1;
		else i = 0;
		for(;i<n-1;i+=2){
			if(buf[i]>buf[i+1]) swap_float(buf+i,buf+i+1);
		}
		t2 = MPI_Wtime();
		CPUtime += t2-t1;
		
		t1 = MPI_Wtime();
		/*** determine the local "0" is odd or even in global ***/
		if(fx){
			// non-blocking send 0-idx to the left
			MPI_Send(buf, 1, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);
			// retrieve the number back
			MPI_Recv(buf, 1, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if(fy){
			// blocking receive a number from the right
			// compare, and then return a number back
			MPI_Recv(&t, 1, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(buf[n-1] > t) swap_float(buf+n-1, &t);
			MPI_Send(&t, 1, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
		}
		t2 = MPI_Wtime();
		COMtime += t2-t1;
		
		t1 = MPI_Wtime();
		/*** odd_phase(); ***/
		// if my 'zero' is even => send
		// if my end is odd => receive & return
		fx = rank && (!(ofs&1)) && (n>0);
		fy = (rank<size-1) && ((ofs+n-1)&1) && (ofs+n<N);

		if(ofs&1) i = 0;
		else i = 1;
		for(;i<n-1;i+=2){
			if(buf[i]>buf[i+1]) swap_float(buf+i,buf+i+1);
		}
		t2 = MPI_Wtime();
		CPUtime += t2-t1;
		
		t1 = MPI_Wtime();
		if(fx){
			// send 0-idx to the left
			MPI_Send(buf, 1, MPI_FLOAT, rank-1, rank, MPI_COMM_WORLD);
			// retrieve the number back
			MPI_Recv(buf, 1, MPI_FLOAT, rank-1, rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if(fy){
			// receive a number and compare
			// then return
			MPI_Recv(&t, 1, MPI_FLOAT, rank+1, rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(buf[n-1] > t) swap_float(buf+n-1, &t);
			MPI_Send(&t, 1, MPI_FLOAT, rank+1, rank, MPI_COMM_WORLD);
		}
		
		MPI_Allreduce(&isSwap,&unsorted,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
		t2 = MPI_Wtime();
		COMtime += t2-t1;
		
	}
	// mpi_write_at ... 
	t1 = MPI_Wtime();
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	MPI_File_write_at(fh, ofs*sizeof(MPI_FLOAT), buf, n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	t2 = MPI_Wtime();
	IOtime += (t2-t1);
	
	t1 = MPI_Wtime();
	free(buf);
	t2 = MPI_Wtime();
	CPUtime += t2-t1;
	
	// collect the timing among all procs
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
