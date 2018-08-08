//#define DLLEXPORT extern "C" __declspec(dllexport)
#define FLOAT_TYPE float
#include<iostream>
using namespace std;

#define NUM_THREADS     6
struct thread_data{
    float *input; 
    int i0,i1,i2,i3; 
    float *filter; 
    int f0, f1, f2, f3, s0, s1, s2, s3; 
    float *output; 
    int o0, o1, o2, o3;
    int M;
};
void *conv2d_c_p(void *threadarg)
{
    struct thread_data *my_data;
    
    my_data = (struct thread_data *) threadarg;
    float *input  =  my_data->input;
    int i0 = my_data -> i0;
    int i1 = my_data -> i1;
    int i2 = my_data -> i2;
    int i3 = my_data -> i3;
    float *filter  =  my_data->filter;
    int f0 = my_data -> f0;
    int f1 = my_data -> f1;
    int f2 = my_data -> f2;
    int f3 = my_data -> f3;
    int s0 = my_data -> s0;
    int s1 = my_data -> s1;
    int s2 = my_data -> s2;
    int s3 = my_data -> s3;
    float *output  =  my_data->output;
    int o0 = my_data -> o0;
    int o1 = my_data -> o1;
    int o2 = my_data -> o2;
    int o3 = my_data -> o3;
    int M = my_data -> M;
    //cout<<"M = "<<M<<endl;
    //cout<<"o0 = "<<o0<<endl;
    for (register int m=M;m<o0;m+=NUM_THREADS){
        //cout<<"m="<<m<<endl;
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                        //mata : 1*i3
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<i3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        //for (int x=0;x<f2;++x){
                            //for (int y=0;y<f3;++y){
                                //std::cout<<matb[x*f3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}

                        //matb : f2*f3
                        register float* matc = output+(m*o1*o2*o3)+i*o2*o3+j*o3;
                            for (register int y=0;y<f3;++y){
                                for (register int z=0;z<f2;++z){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[y]+=mata[z]*matb[z*f3+y];
                                }
                            }
                        
                    }
                }
                
            }
        }
    }
    //pthread_exit(NULL);
    return 0;

}
void *conv2d_c_grad1_p(void *threadarg)
{
    struct thread_data *my_data;
    
    my_data = (struct thread_data *) threadarg;
    float *input  =  my_data->input;
    int i0 = my_data -> i0;
    int i1 = my_data -> i1;
    int i2 = my_data -> i2;
    int i3 = my_data -> i3;
    float *filter  =  my_data->filter;
    int f0 = my_data -> f0;
    int f1 = my_data -> f1;
    int f2 = my_data -> f2;
    int f3 = my_data -> f3;
    int s0 = my_data -> s0;
    int s1 = my_data -> s1;
    int s2 = my_data -> s2;
    int s3 = my_data -> s3;
    float *output  =  my_data->output;
    int o0 = my_data -> o0;
    int o1 = my_data -> o1;
    int o2 = my_data -> o2;
    int o3 = my_data -> o3;
    int M = my_data -> M;
    //cout<<"M = "<<M<<endl;
    //cout<<"o0 = "<<o0<<endl;
    for (register int m=M;m<o0;m+=NUM_THREADS){
        //cout<<"m="<<m<<endl;
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =output+ m*(o1*o2*o3)+(i)*(o2*o3)+(j)*(o3);
                        //mata : 1*o3
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<o3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        //for (int x=0;x<f2;++x){
                        //    for (int y=0;y<f3;++y){
                        //        std::cout<<matb[x*f3+y] <<" ";
                        //    }
                        //    std::cout<<std::endl;
                        //}

                        //matb : f2*f3
                        register float* matc = input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                            for (register int y=0;y<f2;++y){
                                for (register int z=0;z<f3;++z){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[y]+=mata[z]*matb[y*f3+z];
                                }
                            }
                        
                        
                    }
                }
                
            }
        }
    }
    //pthread_exit(NULL);
    return 0;

}
void *conv2d_c_grad2_p(void *threadarg)
{
    struct thread_data *my_data;
    
    my_data = (struct thread_data *) threadarg;
    float *input  =  my_data->input;
    int i0 = my_data -> i0;
    int i1 = my_data -> i1;
    int i2 = my_data -> i2;
    int i3 = my_data -> i3;
    float *filter  =  my_data->filter;
    int f0 = my_data -> f0;
    int f1 = my_data -> f1;
    int f2 = my_data -> f2;
    int f3 = my_data -> f3;
    int s0 = my_data -> s0;
    int s1 = my_data -> s1;
    int s2 = my_data -> s2;
    int s3 = my_data -> s3;
    float *output  =  my_data->output;
    int o0 = my_data -> o0;
    int o1 = my_data -> o1;
    int o2 = my_data -> o2;
    int o3 = my_data -> o3;
    int M = my_data -> M;
    //cout<<"M = "<<M<<endl;
    //cout<<"o0 = "<<o0<<endl;
    for (register int m=M;m<o0;m+=NUM_THREADS){
        //cout<<"m="<<m<<endl;
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                        //mata : i3*1
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<i3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = output+(m*o1*o2*o3)+i*o2*o3+j*o3;
                        //for (int x=0;x<f2;++x){
                            //for (int y=0;y<f3;++y){
                                //std::cout<<matb[x*f3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}

                        //matb : 1*o3
                        register float* matc = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        for (register int x=0;x<i3;++x){
                            for (register int y=0;y<o3;++y){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[x*o3+y]+=mata[x]*matb[y];
                                
                            }
                        }
                        
                    }
                }
                
            }
        }
    }
    //pthread_exit(NULL);
    return 0;

}



extern "C"
int conv2d_c(float *input, int i0,int i1,int i2,int i3, float *filter, int f0,int f1,int f2,int f3, int s0,int s1,int s2,int s3,float *output, int o0,int o1,int o2,int o3){



    pthread_t threads[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    int rc;
    int i;
 
    for( i=0; i < NUM_THREADS; i++ ){
        //cout <<"main() : creating thread, " << i << endl;
        td[i].M = i;
        td[i].input = input;
        td[i].filter = filter;
        td[i].output = output;
        td[i].i0=i0;
        td[i].i1=i1;
        td[i].i2=i2;
        td[i].i3=i3;
        td[i].f0=f0;
        td[i].f1=f1;
        td[i].f2=f2;
        td[i].f3=f3;
        td[i].s0=s0;
        td[i].s1=s1;
        td[i].s2=s2;
        td[i].s3=s3;
        td[i].o0=o0;
        td[i].o1=o1;
        td[i].o2=o2;
        td[i].o3=o3;
        
        //td[i].message = (char*)"This is message";
        rc = pthread_create(&threads[i], NULL,
                          conv2d_c_p, (void *)&td[i]);
        if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            //exit(-1);
        }
    }    
        //cout<<"OK"<<endl;

    //pthread_exit(0);
    void *status[NUM_THREADS];
    for( i=0; i < NUM_THREADS; i++ )  pthread_join(threads[i], &status[i]);
    //cout<<"OK"<<endl;
    return 0;
    for (register int m=0;m<o0;++m){
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                        //mata : 1*i3
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<i3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        //for (int x=0;x<f2;++x){
                            //for (int y=0;y<f3;++y){
                                //std::cout<<matb[x*f3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}

                        //matb : f2*f3
                        register float* matc = output+(m*o1*o2*o3)+i*o2*o3+j*o3;
                            for (register int y=0;y<f3;++y){
                                for (register int z=0;z<f2;++z){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[y]+=mata[z]*matb[z*f3+y];
                                }
                            }
                        
                    }
                }
                
            }
        }
    }
    return 0;
}
extern "C"
int conv2d_c_grad1(float *input, int i0,int i1,int i2,int i3, float *filter, int f0,int f1,int f2,int f3, int s0,int s1,int s2,int s3,float *output, int o0,int o1,int o2,int o3){
    pthread_t threads[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    int rc;
    int i;
 
    for( i=0; i < NUM_THREADS; i++ ){
        //cout <<"main() : creating thread, " << i << endl;
        td[i].M = i;
        td[i].input = input;
        td[i].filter = filter;
        td[i].output = output;
        td[i].i0=i0;
        td[i].i1=i1;
        td[i].i2=i2;
        td[i].i3=i3;
        td[i].f0=f0;
        td[i].f1=f1;
        td[i].f2=f2;
        td[i].f3=f3;
        td[i].s0=s0;
        td[i].s1=s1;
        td[i].s2=s2;
        td[i].s3=s3;
        td[i].o0=o0;
        td[i].o1=o1;
        td[i].o2=o2;
        td[i].o3=o3;
        
        //td[i].message = (char*)"This is message";
        rc = pthread_create(&threads[i], NULL,
                          conv2d_c_grad1_p, (void *)&td[i]);
        if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            //exit(-1);
        }
    }    
        //cout<<"OK"<<endl;

    //pthread_exit(0);
    void *status[NUM_THREADS];
    for( i=0; i < NUM_THREADS; i++ )  pthread_join(threads[i], &status[i]);
    //cout<<"OK"<<endl;
    return 0;
    for (register int m=0;m<o0;++m){
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =output+ m*(o1*o2*o3)+(i)*(o2*o3)+(j)*(o3);
                        //mata : 1*o3
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<o3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        //for (int x=0;x<f2;++x){
                        //    for (int y=0;y<f3;++y){
                        //        std::cout<<matb[x*f3+y] <<" ";
                        //    }
                        //    std::cout<<std::endl;
                        //}

                        //matb : f2*f3
                        register float* matc = input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                            for (register int y=0;y<f2;++y){
                                for (register int z=0;z<f3;++z){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[y]+=mata[z]*matb[y*f3+z];
                                }
                            }
                        
                    }
                }
                
            }
        }
    }
    return 0;
}

extern "C"
int conv2d_c_grad2(float *input, int i0,int i1,int i2,int i3, float *filter, int f0,int f1,int f2,int f3, int s0,int s1,int s2,int s3,float *output, int o0,int o1,int o2,int o3){
    pthread_t threads[NUM_THREADS];
    struct thread_data td[NUM_THREADS];
    int rc;
    int i;
 
    for( i=0; i < NUM_THREADS; i++ ){
        //cout <<"main() : creating thread, " << i << endl;
        td[i].M = i;
        td[i].input = input;
        td[i].filter = filter;
        td[i].output = output;
        td[i].i0=i0;
        td[i].i1=i1;
        td[i].i2=i2;
        td[i].i3=i3;
        td[i].f0=f0;
        td[i].f1=f1;
        td[i].f2=f2;
        td[i].f3=f3;
        td[i].s0=s0;
        td[i].s1=s1;
        td[i].s2=s2;
        td[i].s3=s3;
        td[i].o0=o0;
        td[i].o1=o1;
        td[i].o2=o2;
        td[i].o3=o3;
        
        //td[i].message = (char*)"This is message";
        rc = pthread_create(&threads[i], NULL,
                          conv2d_c_grad2_p, (void *)&td[i]);
        if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            //exit(-1);
        }
    }    
        //cout<<"OK"<<endl;

    //pthread_exit(0);
    void *status[NUM_THREADS];
    for( i=0; i < NUM_THREADS; i++ )  pthread_join(threads[i], &status[i]);
    //cout<<"OK"<<endl;
    return 0;
    for (register int m=0;m<o0;++m){
        for (register int i=0;i<o1;++i){
            for (register int j=0;j<o2;++j){
                for (register int di=0;di<f0;++di){
                    for (register int dj=0;dj<f1;++dj){
                        register float* mata =input+ m*(i1*i2*i3)+(s1*i+di)*(i2*i3)+(s2*j+dj)*(i3);
                        //mata : i3*1
                        //for (int x=0;x<1;++x){
                            //for (int y=0;y<i3;++y){
                                //std::cout<<mata[x*i3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}
                        register float* matb = output+(m*o1*o2*o3)+i*o2*o3+j*o3;
                        //for (int x=0;x<f2;++x){
                            //for (int y=0;y<f3;++y){
                                //std::cout<<matb[x*f3+y] <<" ";
                            //}
                            //std::cout<<std::endl;
                        //}

                        //matb : 1*o3
                        register float* matc = filter+ di*(f1*f2*f3)+dj*(f2*f3);
                        for (register int x=0;x<i3;++x){
                            for (register int y=0;y<o3;++y){
                                    //matc[x][y]+=mata[x][z]*matb[x][y]
                                    matc[x*o3+y]+=mata[x]*matb[y];
                                
                            }
                        }
                    }
                }
                
            }
        }
    }
    return 0;
}


