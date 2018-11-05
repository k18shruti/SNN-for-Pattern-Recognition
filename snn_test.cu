//2 layered neural network with LIF neurons
//computing Vm in parallel, Computing Isyn 
//all-all connectivity between 2 layers
//starting point of reading mnist set by 'start'
//test: Test the trained mnist network on the images of handwritten
 
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

#define w_lat -1.0E-9	//inhibitory lateral

//num of neurons in layer 1 and layer 2
#define p 28
#define q 28
#define N_imgs 10000
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
#define max_epochs 1	     //1 complete presentation of all images
#define decay_LP 0.9802

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

__device__ unsigned char Y_op[N2][M];
__device__ double ci[N1], cil[N2];
__device__ double cis[N1],cils[N2];

//Neuron variables:
__device__ int ref_time1[N1],ref_time2[N2];
__device__ double Vm1[N1],Vm2[N2];

//Correlation variables:
__device__ double norm_obs[N2], norm_des;
__device__ double L_obs[N2][M], sum_obs[N2];
__device__ double L_des[M], sum_des;

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

__global__ void convert2Spks(unsigned char *pix_spks_d)
{
    //int time_period=1;

    for(int i=0; i<256; i++) {
	syn1[i]=0.0; syn1s[i]=0.0;
        for(int l=0; l<M; l++) {
	    //pix_spks[i][l]=0;
	    pix_spks[i][l]=pix_spks_d[i*M+l];
	}
    }

    /*for(int i=0; i<256; i++) {  //all possible pixel values
        if(i>7) {
            time_period = round((1.0/(double)i)/0.1E-3);
            for(int l=time_period; l<M; l+=time_period) {
                pix_spks[i][l]=1;
            }
        }
    }*/

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

//spikes with poisson distributed inter-arrival times
//NOTE: rand fn not allowed inside cuda kernel
//so making this a cpu function
unsigned char pixspks[256*M];

__global__ void pix2spks(int k, int i) 
{
    int tx=threadIdx.x+blockIdx.x*blockDim.x;

    if(tx<(p*q)) {
	img_spks[tx][i]=syn[d_imgs[k][tx+1]][i];
	__syncthreads();
    }
}

/*__global__ void pix2spks(int k)
{
    //Load the spikes trains for the corresponding pixels:
    for(int i=1; i<(p*q+1); i++) {
         for(int l=0; l<M; l++) {
             img_spks[i-1][l]=syn[d_imgs[k][i]][l];
	     //if(img_spks[i-1][l]!=0) printf("img_Spks[%d][%d]=%e\n",i-1,l,img_spks[i-1][l]);
         }
    }

}*/

__device__ unsigned char D_op[M];

__global__ void createDes() {

    for(int i=0; i<M;i++) {
        D_op[i]=0;
    }
    for(int i=100; i<M; i+=35) {
        D_op[i]=1;
    }
    L_des[0]=0.0;

    for(int i=1; i<M; i++) {
	L_des[i]=L_des[i-1]*decay_LP;
	if(D_op[i]==1)
	    L_des[i]+=1;
	sum_des+=L_des[i]*L_des[i];
    }
    norm_des=sqrt(sum_des);
}

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
        ci[tx]=0.0; cis[tx]=0.0;
	cil[ty]=0.0; cils[ty]=0.0;

	Isyn[tx][ty]=0.0; Isyn_lat[ty]=0.0; 
	sum_obs[ty]=0.0; L_obs[ty][0]=0.0;
    }
}

//__global__ void LifKernel1(int i, double *V1, double *I1, unsigned char *spk)
__global__ void LifKernel1(int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double k1, k2;

    if(j<N1) {

	if(ref_time1[j]<i) ref_time1[j]=0;

        k1=(-gL*(Vm1[j]-EL)+I_in[j])/C;
	k2=(-gL*((Vm1[j]+dt*k1)-EL)+I_in[j])/C;

	Vm1[j]+=(dt*(k1+k2)/2)*(ref_time1[j]==0);

	if(Vm1[j]<EL)
	    Vm1[j]=EL;

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

//__global__ void LifKernel2(int i, double *V2, double *I2, unsigned char *spk)
__global__ void LifKernel2(int i)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double k1, k2;

    if(j<N2) {
	if(ref_time2[j]<i) ref_time2[j]=0;

        k1=(-gL*(Vm2[j]-EL)+Isyn_tot[j]+Isyn_lat[j])/C;
	k2=(-gL*((Vm2[j]+dt*k1)-EL)+Isyn_tot[j]+Isyn_lat[j])/C;

	Vm2[j]+=(dt*(k1+k2)/2)*(ref_time2[j]==0);

	if(Vm2[j]<EL)
	    Vm2[j]=EL;

        if(Vm2[j]>=VT) {
	    Vm2[j]=EL;
	    ref_time2[j]=i+round(Rp/dt);
	    Y_op[j][i]=1;
	} else {
	    Y_op[j][i]=0;
	}

	//compute LPF of the output spike trains:
        if(i>0) {
	    L_obs[j][i]=L_obs[j][i-1]*decay_LP;
	    if(Y_op[j][i]==1) {
		L_obs[j][i]+=1;
	    }
	    sum_obs[j]+=L_obs[j][i]*L_obs[j][i];
	}
	if(i==M-1)
	    norm_obs[j]=sqrt(sum_obs[j]);

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
        if(Y_op[ix][i-1]==1) {
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

__global__ void IsynRedKernel(int i)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int tid=threadIdx.x;

    for(unsigned int s=blockDim.x/2;s>0;s>>=1) {
        if(iy<N2 && tid<s && (ix+s)<N1) {
            Isyn[ix][iy]+=Isyn[ix+s][iy];
        }
    __syncthreads();
    }

    if(tid==0 && iy<N2) {
	total_curr[blockIdx.x][iy]=Isyn[ix][iy];
    }
}


__global__ void reduce1(int i) {
    int tx=threadIdx.x+blockIdx.x*blockDim.x;

    if(tx<N2) {
	double total=0.0;
	for(int k=0; k<8; k++) {
	    total+=total_curr[k][tx];
	}
	Isyn_tot[tx]=total;
    }
}

__device__ int match, match_corr;

__global__ void Trained_chk(int k, double *success, double *Corr_acc)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    __shared__ int Tobs[N2];
    __shared__ double corr[N2];
    __shared__ int max, ind;
    __shared__ int max2, ind2;

    if(ix<N2) {
        Tobs[ix]=0; corr[ix]=0.0;
 	double dot_lp=0;
        for(int i=0; i<M; i++) {
	   if(Y_op[ix][i]==1)
	       Tobs[ix]++;

	   //correltion betwn. desired and observed:
	   dot_lp+=L_obs[ix][i]*L_des[i];
        }
	if(norm_obs[ix]!=0) {
	    corr[ix]=dot_lp/(norm_des*norm_obs[ix]);
	}
	__syncthreads();
    }

    //find max in tobs:
    if(ix==0) {
        max=0; ind=0;
        for(int j=0; j<N2; j++) {
	    if(max<Tobs[j]) {
	        max=Tobs[j];
		ind=j;
	    }
	}

	max2=0; Tobs[ind]=0;
	for(int j=0; j<N2; j++) {
	    if(max2<Tobs[j]) {
		ind2=j;
	        max2=Tobs[j];
	    }
	}
	__shared__ double onoff;
	onoff=(double)max/(double)max2;
	*success=0.0;

	printf("img=%d max=%d, ind=%d, max2=%d,ind2=%d, onoff=%f\n",d_imgs[k][0],max,ind,max2, ind2, onoff);

	if(ind==d_imgs[k][0] && max!=0 && onoff>1.0) 
	    match++;

	printf("match=%d\n",match);

	//find max in corr:
	double max_corr=0; 
	int ind_max=0;
	for(int j=0; j<N2; j++) {
	    if(max_corr<corr[j]) {
		max_corr=corr[j];
		ind_max=j;
	    }
	}
	if(ind_max==d_imgs[k][0] && max_corr!=0.0) {
	    match_corr++;
	}
	else
	    printf("k=%d, ouput=%d did not match with label%d with max_corr=%f\n",k,ind_max,d_imgs[k][0],max_corr);

        printf("max_corr=%e, ind_max=%d, label=%d, match_corr=%d\n",max_corr,ind_max,d_imgs[k][0],match_corr);

        if(k==(N_imgs-1)) {
	    *success=(double)match/(double)N_imgs;
	    *Corr_acc=(double)match_corr/(double)N_imgs;
	}
    }

}

__global__ void reset()
{
    match=0; match_corr=0;
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
    double temp;


    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
    double temp;

    if(ix<L && iy<L) {
        temp=0.0;
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
double c_wts[12*Nw*Nw];

//////////storage varialbes///////////
//double V1[N1], V2[N2], I1[N1], I2[N2];
//unsigned char spk1[N1], spk2[N2];
//////////////////////////////////////

int main(int argc, char *argv[])
{
    int i;
    int start=atoi(argv[1]);
    FILE *FWI;	//to load initial wts and store final wts
    FILE *conv_wt;
 
    clock_t t1, t2;
    long elapsed; //time_1pass=0;

    FILE *F_test, *F_lif_spks;

    //for concurrent execution of different kernels:
    cudaStream_t stream2, stream3, stream4, stream5, stream6;
    cudaStream_t stream7, stream8, stream9, stream10, stream11, stream12, stream13;
    cudaStream_t stream14, stream15;

    printf("starting program\n");
    cudaSetDevice(6);    //set the number of GPU you want to use for this simuluation
    
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

    printf("testing the network trained with 60000 MNIST train images\n");

    F_test = fopen("mnist_val.csv","r");            //load the csv files for MNIST dataset
    //F_test = fopen("mnist_test.csv","r");
    if(F_test == NULL)
    {
	perror("Error while opening file mnist_train.csv\n");
	exit(EXIT_FAILURE);
    }

    FWI = fopen("wts_trained_Final.txt","r");        //load the trained weights for the particular epoch
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

    F_lif_spks = fopen("pixels_spks.csv","r");
    if(F_lif_spks == NULL)
    {
	perror("Error while opening file pixels_spks.csv\n");
	exit(EXIT_FAILURE);
    }
 
    printf("Total no. of neurons=%d, no. of synapses to be trained=%d\n",(N1+N2), N1*N2);
    cudaMalloc((void**)&d_imgs_lin,sizeof(int)*(N_imgs*(p*q+1)));
    cudaMalloc((void**)&d_wts,N1*N2*sizeof(double));
    cudaMalloc((void**)&dcwts,12*Nw*Nw*sizeof(double));
    cudaMalloc((void**)&pix_spks_d,256*M*sizeof(unsigned char));

    //Read the initial weights: 
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
            fscanf(F_test,"%d,",&test_set[n][j]);
        }
    }
    fclose(F_test);

    //convert 2D matrix to 1D for transfer to device:
    for(int n=0; n<(N_imgs);n++) {
        for(int j=0;j<(p*q+1);j++) {
            img_lin[n*(p*q+1)+j]=test_set[n][j];
        }
    }

    cudaMemcpy(d_imgs_lin,img_lin,sizeof(int)*(N_imgs*(p*q+1)),cudaMemcpyHostToDevice);

    //call cuda kernel to read in the images:
    img_readKernel<<<1,1>>>(d_imgs_lin);
    cudaDeviceSynchronize();
    cudaFree(d_imgs_lin);
    createDes<<<1,1>>>();

    int NBlks=(N1/Nthrds)+1;

    dim3 dimGrid(82,1,1);
    dim3 dimBlock(100,10,1);

    dim3 grid_syn(NBlks,10,1);
    dim3 block_syn(Nthrds,1,1);

    //convert the pixel values from 0 to 255 into spike trains
    convert2Spks<<<1,1>>>(pix_spks_d);
    cudaDeviceSynchronize();
    cudaFree(pix_spks_d);
    
    dim3 convGrid(1,1,1);
    dim3 convBlks(26,26,1);

    double *d_success, *success, *corr_acc, *h_corr_acc;
    d_success = (double*)malloc(sizeof(double));
    h_corr_acc = (double*)malloc(sizeof(double));
    cudaMalloc((void**)&success,sizeof(double));
    cudaMalloc((void**)&corr_acc,sizeof(double));

    //CPU time required for computation
    t1 = clock();

    int l=0; 				//image index
    for(int n=0; n<max_epochs; n++) {
	reset<<<1,1>>>();
        for(l=0; l<N_imgs; l++) {
            printf("l=%d image %d\n",l,test_set[l][0]);

            //Clear all neuron/synapse variables
	    clear_vars<<<dimGrid,dimBlock>>>();
            cudaDeviceSynchronize();

            for(i=0; i<M; i++) {
                pix2spks<<<1,784>>>(l,i);
		/////////////////////////////////////////////////
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
		////////////////////////////////////////////////////////
		cudaDeviceSynchronize();

                LifKernel1<<<NBlks,Nthrds>>>(i);
		/////////////////////////////////////////////////////////
		cudaDeviceSynchronize();
                SynKernel<<<dimGrid,dimBlock,0,stream14>>>(i);
		Lat_curr<<<1,10,0,stream15>>>(i);
		/////////////////////////////////////////////////////////
		cudaDeviceSynchronize();
                IsynRedKernel<<<grid_syn,block_syn>>>(i);
		reduce1<<<10,1>>>(i);

		//////////////////////////////////////////////////
		cudaDeviceSynchronize();
                LifKernel2<<<1,10>>>(i);
		cudaDeviceSynchronize();
	    }
            //verify if image is correctly identified as per onoff criteria:
            Trained_chk<<<1,10>>>(l,success,corr_acc);
        }		//end of loops over N_imgs

        cudaMemcpy(d_success,success,sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_corr_acc,corr_acc,sizeof(double),cudaMemcpyDeviceToHost);

	printf("success till %d epochs using count=%f\n",(n/2),*d_success);
	printf("Accuracy till %d epochs using correlation=%f\n",(n/2),*h_corr_acc);
    }				//end of n epochs loops
    cudaDeviceSynchronize();

    t2 = clock();
    elapsed = timediff(t1,t2);
    printf("Elapsed time: %ld ms\n", elapsed);
    cudaFree(success);

    cudaDeviceReset();
    return(0);
}
