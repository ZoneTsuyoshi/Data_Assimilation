#Twin experiment
#True parameters
param<-c(10,28,8/3);
init<-c(5,5,5);
time<-15;
dt<-0.01;
obs_sim_sd<-2.0;
obs_inter<-0.2;

#DA parameters
sys_sd<-0.01;
obs_sd<-4.0;
da_init<-c(4.95,5.05,5.05);
nsample<-250;

#calc num of steps
step<-ceiling(time/dt)+1;

# perfect simulation
sim_res<-lor_full(init,param,time,dt);

# observation 
obs_time<-seq(round(obs_inter/dt),step,round(obs_inter/dt));
obs<-as.list(NULL);
## three dim obs/ online generation
# for(i in 1:3){
#   obs_base<-sim_res[[i]][obs_time]+rnorm(length(obs_time),0,obs_sim_sd);
#   obs2<-numeric(step)*NA;
#   obs2<-replace(obs2,obs_time,obs_base);
#   obs<-append(obs,list(obs2))
# }
## one dim obs/ online generation
#  for(i in 1:1){
#    obs_base<-sim_res[[i]][obs_time]+rnorm(length(obs_time),0,obs_sim_sd);
#    obs2<-numeric(step)*NA;
#    obs2<-replace(obs2,obs_time,obs_base);
#    obs<-append(obs,list(obs2))
# }
## observation series is read from data file
obs<-append(obs,list(as.numeric(scan("observation.dat"))))

plot(sim_res[[1]],type="l",ylim=c(-20,20),xlim=c(1,step),xlab="time step",ylab="",main="u / true_state and obs")
par(new=T)
plot(obs[[1]],type="p",ylim=c(-20,20),xlim=c(1,step),xlab="",ylab="")


# Free run
free_run_res<-lor_full(da_init,param,time,dt);
plot(sim_res[[1]],type="l",ylim=c(-20,20),xlim=c(1,step),xlab="time step",ylab="",main="u / true_state and freerun")
par(new=T)
plot(free_run_res[[1]],type="l",lty="dashed",ylim=c(-20,20),xlim=c(1,step),xlab="",ylab="")


# DA by PF
state_list<-lor_pf_init(da_init,0.1,nsample);
da_run_res<-list(numeric(step),numeric(step),numeric(step));
da_run_res[[1]][1]<-mean(mapply(as.matrix,state_list)[1,]);
da_run_res[[2]][1]<-mean(mapply(as.matrix,state_list)[2,]);
da_run_res[[3]][1]<-mean(mapply(as.matrix,state_list)[3,]);
i<-2;
while(i<=step){
  state_list<-lor_pf_pred(state_list,param,dt,sys_sd,nsample);
  if(!is.na(obs[[1]][i])){
    state_list<-lor_pf_filt(state_list,list(obs[[1]][i]),obs_sd,nsample);
#    state_list<-lor_pf_filt(state_list,list(obs[[1]][i],obs[[2]][i],obs[[3]][i]),obs_sd,nsample);
  }
  da_run_res[[1]][i]<-mean(mapply(as.matrix,state_list)[1,]);
  da_run_res[[2]][i]<-mean(mapply(as.matrix,state_list)[2,]);
  da_run_res[[3]][i]<-mean(mapply(as.matrix,state_list)[3,]);
  i<-i+1;
  print(i);
}

plot(sim_res[[1]],type="l",ylim=c(-20,20),xlim=c(1,step),xlab="time step",ylab="",main="u / true_state and DA")
par(new=T)
plot(da_run_res[[1]],type="l",lty="dashed",ylim=c(-20,20),xlim=c(1,step),xlab="",ylab="")
