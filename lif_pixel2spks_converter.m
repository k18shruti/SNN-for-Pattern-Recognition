% Neuron to simulate pixel to spike conversion:
close all;
clear all;
clc;

seed=1957;

rng(seed,'twister');

T=0.1;              % simulation time
dt=0.1e-3;          % time step (0.1ms)
M=round(T/dt);
tref =3e-3;         % refractory period, 3ms

MAX_spks=ceil(T/tref);  % max no. of spikes possible in the sim time

t=0:dt:T-dt;

% Neuron parameters
C=300e-12; gL=30E-9; EL=-70e-3; VT=20e-3; tau = 5e-3;  tau_s=tau/4;
tau_Ld=1e-3; tau_N=10e-3;

decay = exp(-dt/tau); decays = exp(-dt/tau_s);
decay1 = exp(-dt/tau_Ld);

% Input pixels:
N_pixel=256;
pix=[0:N_pixel-1]';
w=1.012e-10; 
Ic=2.7e-9; %min current needed for the neuron to spike for constant input
Vm=zeros(N_pixel,M); Y_spk=zeros(N_pixel,M);
isref_n=zeros(N_pixel,1);
i_in=Ic+ pix*w;
%i_in=2700e-12+pix*w;
for i=1:M
    %compute the neuron potential using RK 2nd order method:
    k1=(1/C)*(i_in - (gL*(Vm(:,i)-EL)));
    k2=(1/C)*(i_in - (gL*(Vm(:,i)+(k1*dt)-EL)));
    Vm(:,i+1) = (Vm(:,i)+(dt*(k1+k2)/2));
    
    Vm(find(t(i)-isref_n<tref),i+1)=EL;
    spind = sign(Vm(:,i+1)-VT);
        
    Vm(Vm(:,i+1)<EL,i+1)=EL;
            
    if(max(spind)>0)
        resetfind_n=find(spind>0);
        isref_n(resetfind_n)=t(i);
        Vm(resetfind_n,i)=VT;
        Y_spk(resetfind_n,i)=1;    
    end
end

f0=figure;
[x,y]=find(Y_spk);
plot(y,x,'o')
title('Neuron Spike Trains')

% spike frequency:
spk_cnt=sum(Y_spk,2);

spk_freq=spk_cnt./T;
figure,plot(spk_freq)
title('Spike freq vs pixel values')
