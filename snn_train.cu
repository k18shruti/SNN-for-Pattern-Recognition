//2 layered neural network with LIF neurons
//computing Vm in parallel, Computing Isyn 
//all-all connectivity between 2 layers
//starting point of reading mnist set by 'start'
 
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#define C 300E-12
#define gL 30E-9
#define VT 20E-3
#define EL -70E-3
#define Rp 3E-3
#define dt 1E-4
#define decay 0.9802
#define decays 0.9231
#define decay1 0.9048
#define WT 5E-9

#define w_lat -1.0E-9		//inhibitory lateral strength

//num of neurons in layer 1 and layer 2
#define p 28
#define q 28
#define N_imgs 50000
#define N2 10		    //no. of neurons in 2nd layer

//Convolution parameters
#define Nw 3
#define L (p-Nw+1)	     //square matrix
#define N1 (12*L*L)	     //no. of neurons in 1st layer
#define Nthrds 1024          //use max no. of threads available per SM
#define Nsyns N1*N2	     //no. of connections
#define T 0.1
#define M 1000

//Training parameters:
#define r 20.0E-11	     //Learning Rate for 100ms 
#define max_epochs 20	     //1 complete presentation of all images

//Variables for image reading
unsigned char *pix_spks_d;
int *d_imgs_lin, img_lin[N_imgs*(p*q+1)];
int test_set[N_imgs][p*q+1];
__device__ int d_imgs[N_imgs][p*q+1];
__device__ double img_spks[p*q][M];
__device__ double syn1[256], syn1s[256], syn[256][M];

__device__ unsigned char in_spk[N1];
__device__ double Isyn[N1][N2], weight[N1][N2];
__device__ double Isyn_tot[N2], Isyn_lat[N2];
__device__ double I_lat[N2];

//weight update variables:
__device__ unsigned char D_op[N2][M];
__device__ signed char err[N2];
__device__ unsigned char Y_op[N2];
__device__ double del_w[N1][N2]; 
__device__ double ci[N1], d_hat[N1];
__device__ double cis[N1], norm_dh; 
__device__ double cil[N2], cils[N2];
__device__ double d_hat_sq[N1];

//Neuron variables:
__device__ int ref_time1[N1],ref_time2[N2];
__device__ double Vm1[N1],Vm2[N2];

////////////CUDA Kernels///////////
__global__ void img_readKernel(int *img)
{
    for(int i=0; i<N_imgs; i++){
        for(int j=0; j<(p*q+1); j++){
	    d_imgs[i][j]=img[i*(p*q+1)+j];
	}
    }
}

__device__ unsigned char pix_spks[256][M];
unsigned char pix_spks_h[256*M];

//uniformly spaced spikes
__global__ void convert2Spks(unsigned char *pix_spks_d)
{

    for(int i=0; i<256; i++) {
	syn1[i]=0.0; syn1s[i]=0.0;
        for(int l=0; l<M; l++) {
	    pix_spks[i][l]=pix_spks_d[i*M+l];
	}
    }

    //Compute the synaptic kernels:
    for(int i=0; i<256; i++) {
        for(int l=0; l<M; l++) {
            syn1[i]*=decay; syn1s[i]*=decays;
            if(pix_spks[i][l]==1) {
                syn1[i]+=1.0;
                syn1s[i]+=1.0;
            }
            syn[i][l]=syn1[i]-syn1s[i];
	}
    }
}

unsigned char pixspks[256*M];

__global__ void pix2spks(int k, int i) 
{
    int tx=threadIdx.x+blockIdx.x*blockDim.x;

    if(tx<(p*q)) {
	img_spks[tx][i]=syn[d_imgs[k][tx+1]][i];
        __syncthreads();
    }
}


__global__ void createDes(int k) {
    int tx=threadIdx.x+blockIdx.x*blockDim.x;

    if(tx<N2) {
	for(int i=0; i<M;i++) {
	    D_op[tx][i]=0;
	}
	if(tx==d_imgs[k][0]) {
	    for(int i=100; i<M; i+=35) 
		D_op[tx][i]=1;
	}
    }
}

/*__global__ void pix2spks(int k)
{
    //Load the spikes trains for the corresponding pixels:
    for(int i=1; i<(p*q+1); i++) {
         for(int l=0; l<M; l++) {
             img_spks[i-1][l]=syn[d_imgs[k][i]][l];
         }
    }

    //Create the desired spike trains:
    for(int j=0; j<N2; j++) {
        for(int i=0; i<M; i++) {
            D_op[j][i]=0;
        }
        if(j==d_imgs[k][0]) {
            //for(int i=100; i<M; i+=50)
            for(int i=100; i<M; i+=35)
                D_op[j][i]=1;
        }
    }
}*/

__device__ double I_in[N1];

__global__ void clear_vars()
{
    int tx=threadIdx.x+blockIdx.x*blockDim.x;
    int ty=threadIdx.y+blockIdx.y*blockDim.y;

    if(tx<N1 && ty<N2) {
    	Vm2[ty]=EL; Isyn_tot[ty]=0.0; 
	ref_time2[ty]=0.0;
	Vm1[tx]=EL; ref_time1[tx]=0.0;
	I_in[tx]=0.0;

	del_w[tx][ty]=0.0; 
	Isyn[tx][ty]=0.0; 
	ci[tx]=0.0; cis[tx]=0.0; d_hat[tx]=0.0;

	cil[ty]=0.0; cils[ty]=0.0;
	Isyn_lat[ty]=0.0;
    }
}

__global__ void LifKernel1(int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double k1, k2;

    if(j<N1) {
	
	if(ref_time1[j]<i) ref_time1[j]=0;

        k1=(-gL*(Vm1[j]-EL)+I_in[j])/C;
	k2=(-gL*((Vm1[j]+dt*k1)-EL)+I_in[j])/C;

	Vm1[j]+=(dt*(k1+k2)/2)*(ref_time1[j]==0);

	if(Vm1[j]<EL) Vm1[j]=EL;

        if(Vm1[j]>=VT) {
	    Vm1[j]=EL;
	    ref_time1[j]=i+round(Rp/dt);
	    in_spk[j]=1;
	} else {
	    in_spk[j]=0;
	}
        __syncthreads();
    }
}

__global__ void LifKernel2(int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double k1, k2;

    if(j<N2) {
	if(ref_time2[j]<i) ref_time2[j]=0;

        k1=(-gL*(Vm2[j]-EL)+Isyn_tot[j]+Isyn_lat[j])/C;
	k2=(-gL*((Vm2[j]+dt*k1)-EL)+Isyn_tot[j]+Isyn_lat[j])/C;

	Vm2[j]+=(dt*(k1+k2)/2)*(ref_time2[j]==0);
	if(Vm2[j]<EL) Vm2[j]=EL;

        if(Vm2[j]>=VT) {
	    Vm2[j]=EL;
	    ref_time2[j]=i+round(Rp/dt);
	    Y_op[j]=1;
	} else {
	    Y_op[j]=0;
	}

	//compute the error:
	err[j]=D_op[j][i]-Y_op[j];

        __syncthreads();
    }
}

//kernels for the total synapses in the network
__global__ void SynKernel(int i)
{
    int ix=blockIdx.x*blockDim.x + threadIdx.x;
    int iy=blockIdx.y*blockDim.y + threadIdx.y;

    if(ix<N1 && iy<N2) {
        if(iy==0) {
   	    ci[ix]=ci[ix]*decay;
	    cis[ix]=cis[ix]*decays;
            if(in_spk[ix]==1) {
	        ci[ix]+=1.0;
	        cis[ix]+=1.0;
	    }
	    d_hat[ix]=d_hat[ix]*decay1 + ((ci[ix]-cis[ix])*dt)/C;
	    d_hat_sq[ix]=d_hat[ix]*d_hat[ix];

	}
	__syncthreads();
        Isyn[ix][iy]=(ci[ix]-cis[ix])*weight[ix][iy];
    }
}

__global__ void Lat_curr(int i)
{
    int ix=blockIdx.x*blockDim.x+threadIdx.x;

    if(ix<N2) {
   	cil[ix]=cil[ix]*decay;
	cils[ix]=cils[ix]*decays;
        if(Y_op[ix]==1) {
	    cil[ix]+=1.0;
	    cils[ix]+=1.0;
	}

	I_lat[ix]=w_lat*(cil[ix]-cils[ix]);

	Isyn_lat[ix]=0;
	for(int k=0; k<N2; k++) {
	    if(k!=ix) {
		Isyn_lat[ix]+=I_lat[k];
	    }
	}
    }
}

__device__ double total_curr[8][N2];
__device__ double total_dhatsq[8];

//optimized version
__global__ void IsynRedKernel(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int tid=threadIdx.x;

    for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
        if(iy<N2 && tid<s && (ix+s)<N1) {
	    Isyn[ix][iy]+=Isyn[ix+s][iy];
	}
	if(iy==0 && tid<s && (ix+s)<N1) {
	    d_hat_sq[ix]+=d_hat_sq[ix+s];
        }
	__syncthreads();
    }

    if(tid==0 && iy<N2) {
	total_curr[blockIdx.x][iy]=Isyn[ix][iy];
	if(iy==0) {
	    total_dhatsq[blockIdx.x]=d_hat_sq[ix];
	}
    }
}

__global__ void reduce1(int i) {
    int tx=threadIdx.x+blockIdx.x*blockDim.x;

    if(tx<N2) {
	double total1=0.0;
	for(int k=0; k<8; k++) {
	    total1+=total_curr[k][tx];
	}
	Isyn_tot[tx]=total1;

        if(tx==0) {
	    double total2=0.0;
	    for(int j=0; j<8; j++) 
		total2+=total_dhatsq[j];

	    norm_dh=sqrt(total2);
	}
    }
}

__global__ void CalcUpdate(int i, double l_rate)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;

    if(ix<N1 && iy<N2) {
        if(norm_dh!=0 && err[iy]!=0) {
	    del_w[ix][iy]+=(err[iy]*l_rate*d_hat[ix]/norm_dh);
	}
	__syncthreads();
    }
}

__global__ void WtUpdt()
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;

    if(ix<N1 && iy<N2) {
        weight[ix][iy]+=del_w[ix][iy];
    }
}

__global__ void cpyWts(double *wts)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;

    if(ix<N1 && iy<N2) {
	wts[ix*N2+iy]=weight[ix][iy];
    }
}

__device__ double w_conv1[Nw][Nw], w_conv2[Nw][Nw];
__device__ double w_conv3[Nw][Nw], w_conv4[Nw][Nw];
__device__ double w_conv5[Nw][Nw], w_conv6[Nw][Nw];
__device__ double w_conv7[Nw][Nw], w_conv8[Nw][Nw];
__device__ double w_conv9[Nw][Nw], w_conv10[Nw][Nw];
__device__ double w_conv11[Nw][Nw], w_conv12[Nw][Nw];

__global__ void initialize2D(double *d_wts, double *c_wts)
{
   for(int i=0; i<N1; i++) {
       for(int j=0; j<N2; j++) {
           weight[i][j]=d_wts[i*N2+j];
	}
    }

    for(int i=0; i<(12*Nw); i++) {
        for(int j=0; j<Nw; j++) {
	    if(i<Nw) {
	    	w_conv1[i][j]=c_wts[i*Nw+j];
	    } else if(i>=Nw && i<(2*Nw)) {
	        w_conv2[i-Nw][j]=c_wts[i*Nw+j];
	    } else if(i>=(2*Nw) && i<(3*Nw)) {
	        w_conv3[i-(2*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(3*Nw) && i<(4*Nw)){
	        w_conv4[i-(3*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(4*Nw) && i<(5*Nw)){
	        w_conv5[i-(4*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(5*Nw) && i<(6*Nw)){
	        w_conv6[i-(5*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(6*Nw) && i<(7*Nw)){
	        w_conv7[i-(6*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(7*Nw) && i<(8*Nw)){
	        w_conv8[i-(7*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(8*Nw) && i<(9*Nw)){
	        w_conv9[i-(8*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(9*Nw) && i<(10*Nw)){
	        w_conv10[i-(9*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(10*Nw) && i<(11*Nw)){
	        w_conv11[i-(10*Nw)][j]=c_wts[i*Nw+j];
	    } else if(i>=(11*Nw) && i<(12*Nw)){
	        w_conv12[i-(11*Nw)][j]=c_wts[i*Nw+j];
	    }
	}
    }
}

__global__ void convKernel1(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv1[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[ix*L+iy]=temp;
    }
}

__global__ void convKernel2(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv2[x][y]*img_spks[(ix+x)*q+iy+y][i]; 
	    }
        }

	I_in[L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel3(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv3[x][y]*img_spks[(ix+x)*q+iy+y][i];
	    }
        }
    
        I_in[2*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel4(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv4[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[3*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel5(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv5[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }
        
	I_in[4*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel6(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv6[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[5*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel7(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv7[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[6*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel8(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv8[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[7*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel9(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv9[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[8*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel10(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv10[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[9*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel11(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv11[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[10*L*L+ix*L+iy]=temp;
    }
}

__global__ void convKernel12(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int x, y;

    if(ix<L && iy<L) {
        double temp=0.0;
	for(x=0; x<Nw; x++) {
	    for(y=0; y<Nw; y++) {
	        temp+=WT*w_conv12[x][y]*img_spks[(ix+x)*q+iy+y][i];
            }
        }

	I_in[11*L*L+ix*L+iy]=temp;
    }
}

long timediff(clock_t t1, clock_t t2) {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}
 
double h_wts[N1*N2], *d_wts;
double *dcwts;
double h_wts_saved[N1*N2], *d_wts_saved;
double c_wts[12*Nw*Nw];

int main(int argc, char *argv[])
{
    int start=atoi(argv[1]);
    FILE *FW, *FWI;	//to load initial wts and store final wts
    FILE *conv_wt;
 
    clock_t t1, t2;
    long elapsed=0;

    FILE *F_train, *F_lif_spks;

    //for concurrent execution of different kernels:
    cudaStream_t stream2, stream3, stream4, stream5, stream6;
    cudaStream_t stream7, stream8, stream9, stream10, stream11, stream12;
    cudaStream_t stream13, stream14,stream15;

    //set the gpu device:
    cudaSetDevice(4);

    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);
    cudaStreamCreate(&stream7);
    cudaStreamCreate(&stream8);
    cudaStreamCreate(&stream9);
    cudaStreamCreate(&stream10);
    cudaStreamCreate(&stream11);
    cudaStreamCreate(&stream12);
    cudaStreamCreate(&stream13);
    cudaStreamCreate(&stream14);
    cudaStreamCreate(&stream15);

    F_lif_spks = fopen("pixels_spks.csv","r");
    if(F_lif_spks == NULL)
    {
	perror("Error while opening file pixels_spks.csv\n");
	exit(EXIT_FAILURE);
    }

    F_train = fopen("mnist_trn.csv","r");
    //F_test = fopen("mnist_test.csv","r");
    if(F_train == NULL)
    {
	perror("Error while opening file mnist_train.csv\n");
	exit(EXIT_FAILURE);
    }

    FWI = fopen("wts_initial.csv","r");
    if(FWI == NULL)
    {
	perror("Error while opening file wts_trained.csv\n");
	exit(EXIT_FAILURE);
    }

    conv_wt=fopen("kernels_3x3.csv","r");
    if(conv_wt==NULL) {
	perror("Error while opening file kernel.csv\n");
    	exit(EXIT_FAILURE);
    }

    printf("Total no. of neurons=%d, no. of synapses to be trained=%d\n",(N1+N2), N1*N2);
    cudaMalloc((void**)&d_imgs_lin,sizeof(int)*(N_imgs*(p*q+1)));
    cudaMalloc((void**)&d_wts,N1*N2*sizeof(double));
    cudaMalloc((void**)&dcwts,12*Nw*Nw*sizeof(double));
    cudaMalloc((void**)&d_wts_saved,N1*N2*sizeof(double));
    cudaMalloc((void**)&pix_spks_d,256*M*sizeof(unsigned char));

    //Read the initial weights: 
    //printf("Reading final trained weights from file\n");
    for(int i=0; i<N1; i++) {
        for(int j=0; j<N2; j++) {
	    fscanf(FWI,"%lf,",&h_wts[i*N2+j]);
	}
    }
    fclose(FWI);

    for(int i=0; i<(12*Nw); i++) {
        for(int j=0; j<Nw; j++) {
	    fscanf(conv_wt,"%lf,",&c_wts[i*Nw+j]);
	}
    }
    fclose(conv_wt);

    for(int i=0; i<(256); i++) {
    	for(int j=0; j<M; j++) {
	    fscanf(F_lif_spks,"%d,",&pix_spks_h[i*M+j]);
	}
    }
    fclose(F_lif_spks);

    cudaMemcpy(pix_spks_d,pix_spks_h,256*M*sizeof(unsigned char),cudaMemcpyHostToDevice);
    cudaMemcpy(d_wts,h_wts,N1*N2*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dcwts,c_wts,12*Nw*Nw*sizeof(double),cudaMemcpyHostToDevice);

    initialize2D<<<1,1>>>(d_wts,dcwts);
    cudaDeviceSynchronize();
    cudaFree(d_wts);
    cudaFree(dcwts);

    //Read the images from file:
    for(int n=0;n<N_imgs;n++) {
        for(int j=0;j<(p*q+1);j++) {
            fscanf(F_train,"%d,",&test_set[n][j]);
        }
    }
    fclose(F_train);

    //convert 2D matrix to 1D for transfer to device:
    for(int n=0; n<(N_imgs);n++) {
        for(int j=0;j<(p*q+1);j++) {
            img_lin[n*(p*q+1)+j]=test_set[n][j];
        }
    }

    cudaMemcpy(d_imgs_lin,img_lin,sizeof(int)*(N_imgs*(p*q+1)),cudaMemcpyHostToDevice);

    //call cuda kernel to read in the images:
    img_readKernel<<<1,1>>>(d_imgs_lin);
    cudaThreadSynchronize();
    cudaFree(d_imgs_lin);

    int NBlks=(N1/Nthrds)+1;
    //printf("blocks=%d, threads=%d\n",NBlks,Nthrds);

    dim3 dimGrid(82,1,1);
    dim3 dimBlock(100,10,1);

    dim3 grid_syn(NBlks,10,1);
    dim3 block_syn(Nthrds,1,1);

    //convert pixel values 0 to 255 into spike trains
    convert2Spks<<<1,1>>>(pix_spks_d);
    cudaDeviceSynchronize();
    cudaFree(pix_spks_d);
    
    dim3 convGrid(1,1,1);
    dim3 convBlks(26,26,1);

    double learn=r;
    //CPU time required for computation
    t1 = clock();
           
    int l=0;				//image index
    for(int n=0; n<max_epochs; n++) {
        printf("Epoch=%d\n",n);

	if(n<3) learn=r;
	else if(n>=3 && n<6) learn=r/2;
	else if(n>=6 && n<9) learn=r/4;
	else if(n>=9 && n<12) learn=r/8;
	else if(n>=12 && n<15) learn=r/16;
	else if(n>=15 && n<18) learn=r/32;
	else learn=r/64;

        for(l=0; l<N_imgs; l++) {
            printf("l=%d image %d\n",l,test_set[l][0]);

	    createDes<<<1,10>>>(l);

	    clear_vars<<<dimGrid,dimBlock>>>();
            cudaDeviceSynchronize();

            //simulate for all time steps
            for(int i=0; i<M; i++) {
                pix2spks<<<1,784>>>(l,i);
		///////////////////////////////////////////////////
		cudaDeviceSynchronize();
                convKernel1<<<convGrid,convBlks,0,stream2>>>(i);
                convKernel2<<<convGrid,convBlks,0,stream3>>>(i);
                convKernel3<<<convGrid,convBlks,0,stream4>>>(i);
                convKernel4<<<convGrid,convBlks,0,stream5>>>(i);
                convKernel5<<<convGrid,convBlks,0,stream6>>>(i);
                convKernel6<<<convGrid,convBlks,0,stream7>>>(i);
                convKernel7<<<convGrid,convBlks,0,stream8>>>(i);
                convKernel8<<<convGrid,convBlks,0,stream9>>>(i);
                convKernel9<<<convGrid,convBlks,0,stream10>>>(i);
                convKernel10<<<convGrid,convBlks,0,stream11>>>(i);
                convKernel11<<<convGrid,convBlks,0,stream12>>>(i);
                convKernel12<<<convGrid,convBlks,0,stream13>>>(i);
		/////////////////////////////////////////////////////
		cudaDeviceSynchronize();

                LifKernel1<<<NBlks,Nthrds>>>(i);
		////////////////////////////////////////////////
		cudaDeviceSynchronize();
                SynKernel<<<dimGrid,dimBlock,0,stream14>>>(i);
		Lat_curr<<<1,10,0,stream15>>>(i);
		////////////////////////////////////////////////

                cudaDeviceSynchronize();
                IsynRedKernel<<<grid_syn,block_syn>>>(i);
	        reduce1<<<10,1>>>(i);

		///////////////////////////////////////////////
                cudaDeviceSynchronize();
                LifKernel2<<<1,10>>>(i);
		////////////////////////////////////////////////////
                cudaDeviceSynchronize();
		CalcUpdate<<<dimGrid,dimBlock>>>(i,learn);
	    }

//////////////////////////////////////////////////////////////////////////////
	    WtUpdt<<<dimGrid,dimBlock>>>();
            //cudaDeviceSynchronize();
        }		//end of loops over N_imgs

        t2 = clock();
        elapsed += timediff(t1,t2);
        printf("Elapsed time: %ld ms\n", elapsed);

	char wts_file[25];
	sprintf(wts_file,"wts_gpu%d.txt",n);

	if((FW=fopen(wts_file,"w"))==NULL) {
	    printf("Failed to open file wts_gpu_sample%d.txt\n",n);
        }

	printf("Copying the trained weights to the host for analysis\n");
    	cpyWts<<<dimGrid,dimBlock>>>(d_wts_saved);
	cudaMemcpy(h_wts_saved,d_wts_saved,N1*N2*sizeof(double), cudaMemcpyDeviceToHost);

	//save the weights in a file:
    	for(int i=0; i<N1; i++) {
    	    for(int j=0; j<N2; j++) {
        	fprintf(FW,"%0.14f,",h_wts_saved[i*N2+j]);
            }
            fprintf(FW,"\n");
    	}
	fclose(FW);
    }				//end of n epochs loops

    cudaFree(d_wts_saved);
 
    cudaDeviceReset();
    return(0);
}
