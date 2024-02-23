#include<mpi.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

#define K 7
#define D 16
#define epoch 10000
#define N 4000

typedef struct{
    int name;
    int type;
    int characters[D];
}animal;

char idx2name[N][20];

int loadData(char *filepath,animal *data){
    FILE *file=fopen(filepath,"r");
    if(!file){
        printf("fail to open file %s!\n",filepath);
        return -1;
    }

    int datanum;
    char line[100];

    while(fscanf(file,"%s",line)!=EOF){
        char *token=strtok(line,",");
        if(token!=NULL){
            strcpy(idx2name[datanum],token);
        }
        data[datanum].name=datanum;
        for(int i=0;i<D;i++){
            token=strtok(NULL,",");
            if(token!=NULL){
                data[datanum].characters[i]=atoi(token);
            }
        }
        token=strtok(NULL,",");
        if(token!=NULL){
            data[datanum].type=atoi(token);
        }
        datanum++;
    }

    return datanum;
}

double distance(int *character,double *center,int dimension){
    double dis=0;
    for(int i=0;i<dimension;i++){
        dis+=(character[i]*1.0-center[i])*(character[i]*1.0-center[i]);
    }
    dis=sqrt(dis);
    return dis;
}

void cluster(animal *data,int dataSize,double data_center[K][D],double new_data_center[K][D],int cnt[K]){
    for(int i=0;i<dataSize;i++){
        double min_dis=1000;
        int clusterId;
        for(int j=0;j<K;j++){
            double dis=distance(data[i].characters,data_center[j],D);
            if(dis<min_dis){
                min_dis=dis;
                clusterId=j;
            }
            //printf("%d,%lf\n",j,dis);
        }

        for(int j=0;j<D;j++){
            new_data_center[clusterId][j]+=data[i].characters[j];
        }
        cnt[clusterId]++;
        data[i].type=clusterId;
    }
}
int main(){
    animal data[N];
    int datanum;
    double cluster_center[K][D];
    int total_cnt[K];
    memset(cluster_center,0,sizeof(cluster_center));

    int my_rank,comm_sz;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Barrier(MPI_COMM_WORLD);

    double start,end,elapsed;
    start=MPI_Wtime();
    
    //进程0读取数据，同时告知每个进程，它需要处理的数据量
    if(my_rank==0){
        datanum=loadData("zoo/zoo.data",data);

        for(int i=1;i<comm_sz;i++){
            int nums=datanum/(comm_sz-1);
            int a=(i-1)*nums;
            int b=i==comm_sz-1?datanum:i*nums;
            int sendNum=b-a;
            MPI_Send(&sendNum,1,MPI_INT,i,0,MPI_COMM_WORLD);
        }
    }else{
        MPI_Recv(&datanum,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    //进程0向其他进程分发数据
    if(my_rank==0){
        for(int i=1;i<comm_sz;i++){
            int nums=datanum/(comm_sz-1);
            int a=(i-1)*nums;
            int b=i==comm_sz-1?datanum:i*nums;
            MPI_Send((void *)(data+a),sizeof(animal)*(b-a),MPI_BYTE,i,0,MPI_COMM_WORLD);
        }
    }else{
        MPI_Recv(data,sizeof(animal)*datanum,MPI_BYTE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }

    //进程0产生随机中心点
    if(my_rank==0){
        int visit[N];
        memset(visit,0,sizeof(visit));

        //选择K个不同的数据点作为初始中心点

        srand((unsigned int)(time(NULL)));
        int i=0;
        while(i<K){
            int idx=rand()%datanum;
            if(!visit[idx]){
                for(int j=0;j<D;j++){
                    cluster_center[i][j]=data[idx].characters[j];
                }
                visit[idx]=1;
                i++;
            }
        }
    }

    //将初始聚类中心发送给其他进程
    MPI_Bcast(cluster_center,K*D,MPI_DOUBLE,0,MPI_COMM_WORLD);

    double local_cluster_center[K][D];
    int local_cnt[K];
    for(int round=0;round<epoch;round++){
        memset(local_cluster_center,0,sizeof(local_cluster_center));
        memset(local_cnt,0,sizeof(local_cnt));
        
        if(my_rank!=0){
            cluster(data,datanum,cluster_center,local_cluster_center,local_cnt);
        }

        memset(cluster_center,0,sizeof(cluster_center));
        memset(total_cnt,0,sizeof(total_cnt));

        MPI_Reduce(local_cluster_center,cluster_center,K*D,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(local_cnt,total_cnt,K,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

        //计算新的聚类中心
        if(my_rank==0){
            for(int i=0;i<K;i++){
                for(int j=0;j<D;j++){
                    if(total_cnt[i]!=0){
                        cluster_center[i][j]/=total_cnt[i];
                    }
                }
            }
        }

        //将计算后的新的聚类中心发送给其他进程
        MPI_Bcast(cluster_center,K*D,MPI_DOUBLE,0,MPI_COMM_WORLD);
        //继续下一轮迭代
    }

    //完成迭代之后，收集数据
    if(my_rank!=0){
        int buf[datanum*2];
        for(int i=0;i<datanum;i++){
            buf[i*2]=data[i].name;
            buf[i*2+1]=data[i].type;
        }
        MPI_Send(buf,datanum*2,MPI_INT,0,0,MPI_COMM_WORLD);
    }else{
        int buf[datanum*2];
        for(int i=1;i<comm_sz;i++){
            int nums=datanum/(comm_sz-1);
            int a=(i-1)*nums;
            int b=i==comm_sz-1?datanum:i*nums;
            MPI_Recv((void *)(&buf[a*2]),(b-a)*2,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }

        int cluster[K][N];
        int clusternum[N];
        memset(clusternum,0,sizeof(clusternum));
        for(int i=0;i<datanum;i++){
            int name=buf[i*2];
            int type=buf[i*2+1];
            cluster[type][clusternum[type]++]=name;
        }

        FILE *output=fopen("kmeans_result_mpi.txt","w");
        for(int i=0;i<K;i++){
            fprintf(output,"cluster %d:\n",i);
            for(int j=0;j<clusternum[i];j++){
                fprintf(output,"%s\n",idx2name[cluster[i][j]]);
            }
        }
        fclose(output);
    }

    end=MPI_Wtime();
    elapsed=end-start;
    
    if(my_rank==0){
        printf("elapsed time:%lf\n",elapsed);
    }
    MPI_Finalize();
    return 0;
}