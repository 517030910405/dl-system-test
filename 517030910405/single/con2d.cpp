//#define DLLEXPORT extern "C" __declspec(dllexport)
#define FLOAT_TYPE float
#include<iostream>
extern "C"
int conv2d_c(float *input, int i0,int i1,int i2,int i3, float *filter, int f0,int f1,int f2,int f3, int s0,int s1,int s2,int s3,float *output, int o0,int o1,int o2,int o3){
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


