lor_update<-function(state,param,dt){
  u<-state[1];
  v<-state[2];
  w<-state[3];
  p<-param[1];
  r<-param[2];
  b<-param[3];
  state[1]<- u+(p*v-p*u)*dt;
  state[2]<- v+(-u*w+r*u-v)*dt;
  state[3]<- w+(u*v-b*w)*dt;
  return(state);  
}

lor_full<-function(init,param,time,dt){
  t<-0;
  state<-init;
  l<-ceiling(time/dt)+1;
  u<-numeric(l);
  v<-numeric(l);
  w<-numeric(l);
  u[1]<-init[1];
  v[1]<-init[2];
  w[1]<-init[3];
  i<-2;
  while(i<=l){
    state<-lor_update(state,param,dt);
    u[i]<-state[1];
    v[i]<-state[2];
    w[i]<-state[3];    
    i<-i+1;
  }
  return(list(u,v,w));
}

# initial
lor_pf_init <- function(da_init,sim_sd,nsample){
  state_list<-list(rnorm(length(da_init),da_init,sim_sd));
  for(i in 2:nsample){
    state_list<-append(state_list,list(rnorm(length(da_init),da_init,sim_sd)));
  }
  return(state_list);
}



# System model
lor_pf_pred <- function(state_list,param,dt,sim_sd,nsample){
  vec_size<-length(state_list[[1]]);
  for(i in 1:nsample){
  state_list[[i]]<-lor_update(state_list[[i]],param,dt)+rnorm(1,numeric(vec_size),numeric(vec_size)+sim_sd);  # add same realization of sys noise to u,v,w
#  state_list[[i]]<-lor_update(state_list[[i]],param,dt)+c(rnorm(1,numeric(vec_size),numeric(vec_size)+sim_sd),0,0);  # add sys noise to u
#  state_list[[i]]<-lor_update(state_list[[i]],param,dt)+rnorm(3,numeric(vec_size),numeric(vec_size)+sim_sd); # add independent realizations of sys noise to u,v,w
  }
  return(state_list);
}

# Filtering
lor_pf_filt <- function(state_list,obs,obs_sd,nsample){
  likelihood<-numeric(nsample)+1;
  print(obs);
  for(i in 1:nsample){
    for(j in 1:1){
#      for(j in 1:1){
      likelihood[i]<-likelihood[i]*dnorm(state_list[[i]][j]-obs[[j]],sd=obs_sd)   # 各粒子の尤度計算
    }
    print(likelihood[i]);
  }
  perm<-sample(1:nsample,nsample,replace=TRUE,prob=likelihood)  # リサンプリングされる粒子番号を決める
  print(perm);
  new_list<-as.list(NULL);
  for(i in 1:nsample){
    new_list<-append(new_list,list(state_list[[perm[i]]]));
  }
  print(new_list);
  return(new_list);
}

