#include<omp.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include<sys/time.h>
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
    #pragma omp parallel for 
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
int main(int argc,char *argv[]){
    int thread_count=atoi(argv[1]);
    omp_set_num_threads(thread_count);
    animal data[N];
    int datanum=loadData("zoo/zoo.data",data);
    double cluster_center[K][D];
    int total_cnt[K];

    struct timeval start,end;
    double elapsed;
    gettimeofday(&start,NULL);

    //产生随机中心点
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

    double local_cluster_center[K][D];
    int local_cnt[K];
    for(int round=0;round<epoch;round++){
        memset(local_cluster_center,0,sizeof(local_cluster_center));
        memset(local_cnt,0,sizeof(local_cnt));
        
        cluster(data,datanum,cluster_center,local_cluster_center,local_cnt);

        memcpy(cluster_center,local_cluster_center,sizeof(local_cluster_center));
        memcpy(total_cnt,local_cnt,sizeof(local_cnt));
        
        //计算新的聚类中心
        #pragma omp parallel for
        for(int i=0;i<K*D;i++){
            if(total_cnt[i/D]!=0){
                cluster_center[i/D][i%D]/=total_cnt[i/D];
            }
        }

        //继续下一轮迭代
    }

    //完成迭代之后，收集数据
    int cluster[K][N];
    int clusternum[N];
    memset(clusternum,0,sizeof(clusternum));
    for(int i=0;i<datanum;i++){
        cluster[data[i].type][clusternum[data[i].type]++]=data[i].name;
    }

    FILE *output=fopen("kmeans_result_omp.txt","w");
    for(int i=0;i<K;i++){
        fprintf(output,"cluster %d:\n",i);
        #pragma omp parallel for
        for(int j=0;j<clusternum[i];j++){
            fprintf(output,"%s\n",idx2name[cluster[i][j]]);
        }
    }
    fclose(output);

    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("elapsed time:%lf\n",elapsed);
    return 0;
}