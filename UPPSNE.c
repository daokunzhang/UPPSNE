// User profile preserving social network embedding

//  The UPPSNE.c code was built upon the word2vec.c from https://code.google.com/archive/p/word2vec/

//  Modifications Copyright 2018 <daokunzhang2015@gmail.com>
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

const long long node_context_hash_size = 1000000000; // Maximum 1G node context pairs

struct node_context
{
    long long cn;
    long long source, target;
};

struct node_neighbor
{
    long long node;
    long long neighbor_size;
    long long *neighbors;
};

struct node_content
{
    long long node;
    long long content_size;
    long long *pos;
    double *vals;
};

struct network
{
    long long node_num;
    long long attribute_num;
    struct node_neighbor *node_neighbors;
    struct node_content *node_contents;
};

char graph_file[MAX_STRING], emb_file[MAX_STRING], time_file[MAX_STRING];

struct network graph;

struct node_context *node_context_list;
long long node_context_list_size = 0, node_context_list_max_size = 1000000;
long long *node_context_hash;
long long *node_freq;

long long window_size = 10, walk_num = 40, walk_length = 100;
long long layer1_size = 128;
double alpha = 0.025, starting_alpha;
double *syn0, *syn1neg, *expTable;
clock_t start, finish;

long long negative = 5;
const long long table_size = 1e8;
long long *table;

// Parameters for node context pair sampling
long long *node_context_alias;
double *node_context_prob;

long long total_samples = 100;

long long GetNodeContextHash(long long node, long long context)
{
    long long hash = node * graph.node_num + context;
    hash = hash % node_context_hash_size;
    return hash;
}

long long SearchNodeContextPair(long long node, long long context)
{
    long long hash = GetNodeContextHash(node, context);
    long long hash_iter = 0;
    long long cur_node, cur_context;
    while(1)
    {
        if (node_context_hash[hash] == -1) return -1;
        cur_node = node_context_list[node_context_hash[hash]].source;
        cur_context = node_context_list[node_context_hash[hash]].target;
        if (cur_node == node && cur_context == context) return node_context_hash[hash];
        hash = (hash + 1) % node_context_hash_size;
        hash_iter++;
        if (hash_iter >= node_context_hash_size)
        {
            printf("The node context hash table is full!\n");
            exit(1);
        }
    }
    return -1;
}

//Add node context pair to the node context list
long long AddNodeContextToList(long long node, long long context)
{
    long long hash;
    long long hash_iter = 0;
    node_context_list[node_context_list_size].source = node;
    node_context_list[node_context_list_size].target = context;
    node_context_list[node_context_list_size].cn = 1;
    node_context_list_size++;
    if (node_context_list_size >= node_context_list_max_size)
    {
        node_context_list_max_size += 100 * graph.node_num;
        node_context_list = (struct node_context *)
                            realloc(node_context_list, node_context_list_max_size * sizeof(struct node_context));
    }
    hash = GetNodeContextHash(node, context);
    while (node_context_hash[hash] != -1)
    {
        hash = (hash + 1) % node_context_hash_size;
        hash_iter++;
        if (hash_iter >= node_context_hash_size)
        {
            printf("The node context hash table is full!\n");
            exit(1);
        }
    }
    node_context_hash[hash] = node_context_list_size - 1;
    return node_context_list_size - 1;
}

void ReadGraph()
{
    FILE *fp;
    long long i, j, k, l;
    fp = fopen(graph_file, "r");
    fscanf(fp, "%lld%lld", &graph.node_num, &graph.attribute_num);
    graph.node_neighbors = (struct node_neighbor *)malloc(graph.node_num*sizeof(struct node_neighbor));
    graph.node_contents = (struct node_content *)malloc(graph.node_num*sizeof(struct node_content));
    node_freq = (long long *)malloc(graph.node_num*sizeof(long long));
    for (i = 0; i < graph.node_num; i++) node_freq[i] = 0;
    for (i = 0; i < graph.node_num; i++)
    {
        fscanf(fp, "%lld", &k);
        graph.node_neighbors[k].node = k;
        graph.node_contents[k].node = k;
        fscanf(fp, "%lld", &l);
        graph.node_neighbors[k].neighbor_size = l;
        graph.node_neighbors[k].neighbors = (long long *)malloc(l * sizeof(long long));
        for (j = 0; j < l; j++)
            fscanf(fp, "%lld", &graph.node_neighbors[k].neighbors[j]);
        fscanf(fp, "%lld", &l);
        graph.node_contents[k].content_size = l;
        graph.node_contents[k].pos = (long long *)malloc(l * sizeof(long long));
        graph.node_contents[k].vals = (double *)malloc(l * sizeof(double));
        for (j = 0; j < l; j++)
            fscanf(fp, "%lld%lf", &graph.node_contents[k].pos[j], &graph.node_contents[k].vals[j]);
    }
    fclose(fp);
}

void InitUnigramTable()
{
    long long a, i;
    double train_pow = 0.0, d1, power = 0.75;
    table = (long long *)malloc(table_size * sizeof(long long));
    for (a = 0; a < graph.node_num; a++) train_pow += pow(node_freq[a], power);
    i = 0;
    d1 = pow(node_freq[0], power) / train_pow;
    for (a = 0; a < table_size; a++)
    {
        table[a] = i;
        if (a / (double)table_size > d1)
        {
            i++;
            d1 += pow(node_freq[i], power) / train_pow;
        }
        if (i >= graph.node_num) i = graph.node_num - 1;
    }
}

void RandomWalk()
{
    long long i, j, k, r;
    long long cur_node, node_context_pos;
    long long *rand_walk_nodes = (long long *) malloc(walk_length * sizeof(long long));
    srand((unsigned)time(NULL));
    for (i = 0; i < walk_num; i++)
    {
        for (j = 0; j < graph.node_num; j++)
        {
            cur_node = j;
            node_freq[cur_node]++;
            rand_walk_nodes[0] = j;
            for (k = 1; k < walk_length; k++)
            {
                if(graph.node_neighbors[cur_node].neighbor_size==0)
                    break;
                cur_node = graph.node_neighbors[cur_node].neighbors[rand()%graph.node_neighbors[cur_node].neighbor_size];
                node_freq[cur_node]++;
                rand_walk_nodes[k] = cur_node;
                for (r = 1; r <= window_size; r++)
                {
                    if (k - r < 0) continue;
                    node_context_pos = SearchNodeContextPair(rand_walk_nodes[k-r], rand_walk_nodes[k]);
                    if (node_context_pos == -1)
                        AddNodeContextToList(rand_walk_nodes[k-r], rand_walk_nodes[k]);
                    else
                        node_context_list[node_context_pos].cn++;
                    node_context_pos = SearchNodeContextPair(rand_walk_nodes[k], rand_walk_nodes[k-r]);
                    if (node_context_pos == -1)
                        AddNodeContextToList(rand_walk_nodes[k], rand_walk_nodes[k-r]);
                    else
                        node_context_list[node_context_pos].cn++;
                }
            }
        }
    }
}

void InitNet()
{
    long long a, b;
    unsigned long long next_random = 1;
    syn0 = (double *)malloc(graph.attribute_num * layer1_size * sizeof(double));
    for (a = 0; a < layer1_size; a++)
        for (b = 0; b < graph.attribute_num; b++)
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * graph.attribute_num + b] = ((next_random & 0xFFFF) / ((double)65536) - 0.5) / layer1_size;
        }
    syn1neg = (double *)malloc(graph.node_num * layer1_size * 2 * sizeof(double));
    for (a = 0; a < graph.node_num; a++)
        for (b = 0; b < layer1_size * 2; b++)
            syn1neg[a * layer1_size * 2 + b] = 0;
    InitUnigramTable();
}

/* The alias sampling algorithm, which is used to sample an node context pair in O(1) time. */
void InitNodeContextAliasTable()
{
    long long k;
    double sum = 0;
    long long cur_small_block, cur_large_block;
    long long num_small_block = 0, num_large_block = 0;
    double *norm_prob;
    long long *large_block;
    long long *small_block;
    node_context_alias = (long long *)malloc(node_context_list_size * sizeof(long long));
    node_context_prob = (double *)malloc(node_context_list_size * sizeof(double));
    norm_prob = (double*)malloc(node_context_list_size * sizeof(double));
    large_block = (long long*)malloc(node_context_list_size * sizeof(long long));
    small_block = (long long*)malloc(node_context_list_size * sizeof(long long));
    for (k = 0; k != node_context_list_size; k++) sum += node_context_list[k].cn;
    for (k = 0; k != node_context_list_size; k++) norm_prob[k] = (double)node_context_list[k].cn * node_context_list_size / sum;
    for (k = node_context_list_size - 1; k >= 0; k--)
    {
        if (norm_prob[k] < 1)
            small_block[num_small_block++] = k;
        else
            large_block[num_large_block++] = k;
    }
    while (num_small_block && num_large_block)
    {
        cur_small_block = small_block[--num_small_block];
        cur_large_block = large_block[--num_large_block];
        node_context_prob[cur_small_block] = norm_prob[cur_small_block];
        node_context_alias[cur_small_block] = cur_large_block;
        norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1;
        if (norm_prob[cur_large_block] < 1)
            small_block[num_small_block++] = cur_large_block;
        else
            large_block[num_large_block++] = cur_large_block;
    }
    while (num_large_block) node_context_prob[large_block[--num_large_block]] = 1;
    while (num_small_block) node_context_prob[small_block[--num_small_block]] = 1;
    free(norm_prob);
    free(small_block);
    free(large_block);
}

long long SampleANodeContextPair(double rand_value1, double rand_value2)
{
    long long k = node_context_list_size * rand_value1;
    return rand_value2 < node_context_prob[k] ? k : node_context_alias[k];
}

void TrainModel()
{
    long long a, b, c, d;
    long long node, context, cur_pair;
    long long count = 0, last_count = 0;
    long long l1, l2, target, label;
    unsigned long long next_random = 1;
    double rand_num1, rand_num2;
    double f, g;
    double *neuron = (double *)calloc(layer1_size * 2, sizeof(double));

    InitNet();
    starting_alpha = alpha;
    printf("Training file: %s\n", graph_file);
    printf("Samples: %lldM\n", total_samples / 1000000);
    printf("Dimension: %lld\n", layer1_size * 2);
    printf("Initial Alpha: %f\n", alpha);

    srand((unsigned)time(NULL));
    if (node_context_list_size > (long long)RAND_MAX + 1)
        printf("Warning: RAND_MAX is not large enough to guarantee the correctness of node context pair sampling!\n");
    while (1)
    {
        if (count >= total_samples) break;
        if (count - last_count > 10000)
        {
            last_count = count;
            printf("Alpha: %f Progress %.3lf%%%c", alpha, (double)count / (double)(total_samples + 1) * 100, 13);
            fflush(stdout);
            alpha = starting_alpha * (1 - (double)count / (double)(total_samples + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }
        rand_num1 = rand() / (RAND_MAX * 1.0 + 1);
        rand_num2 = rand() / (RAND_MAX * 1.0 + 1);
        cur_pair = SampleANodeContextPair(rand_num1, rand_num2);
        node = node_context_list[cur_pair].source;
        context = node_context_list[cur_pair].target;
        for (a = 0; a < layer1_size; a++)
        {
            neuron[a] = 0;
            l1 = a * graph.attribute_num;
            for (b = 0; b < graph.node_contents[node].content_size; b++)
            {
                c = l1 + graph.node_contents[node].pos[b];
                neuron[a] += syn0[c] * graph.node_contents[node].vals[b];
            }
            neuron[a + layer1_size] = neuron[a];
            neuron[a] = 1 / sqrt(layer1_size) * cos(neuron[a]);
            neuron[a + layer1_size] = 1 / sqrt(layer1_size) * sin(neuron[a + layer1_size]);
        }
        for (d = 0; d < negative + 1; d++)
        {
            if (d == 0)
            {
                target = context;
                label = 1;
            }
            else
            {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                target = table[(next_random >> 16) % table_size];
                if (target == context) continue;
                label = 0;
            }
            l2 = target * layer1_size * 2;
            f = 0;
            for (c = 0; c < layer1_size * 2; c++) f += neuron[c] * syn1neg[c + l2];
            if (f > MAX_EXP) f = 1;
            else if (f < -MAX_EXP) f = 0;
            else f = expTable[(int)((f + MAX_EXP) * EXP_TABLE_SIZE / MAX_EXP / 2)];
            g = (label - f) * alpha;
            for (c = 0; c < layer1_size; c++)
            {
                l1 = c * graph.attribute_num;
                for (a = 0; a < graph.node_contents[node].content_size; a++)
                {
                    b = l1 + graph.node_contents[node].pos[a];
                    syn0[b] -= g * syn1neg[c + l2] * neuron[c + layer1_size] * graph.node_contents[node].vals[a];
                    syn0[b] += g * syn1neg[c + layer1_size + l2] * neuron[c] * graph.node_contents[node].vals[a];
                }
            }
            for (c = 0; c < layer1_size * 2; c++) syn1neg[c + l2] += g * neuron[c];
        }
        count++;
    }
}

void Output()
{
    long long a, b, c, d;
    long long l1;
    FILE *fp = fopen(emb_file, "w");
    double *neuron = (double *)calloc(layer1_size * 2, sizeof(double));
    for (a = 0; a < graph.node_num; a++)
    {
        for (b = 0; b < layer1_size; b++)
        {
            neuron[b] = 0;
            l1 = b * graph.attribute_num;
            for (c = 0; c < graph.node_contents[a].content_size; c++)
            {
                d = l1 + graph.node_contents[a].pos[c];
                neuron[b] += syn0[d] * graph.node_contents[a].vals[c];
            }
            neuron[b + layer1_size] = neuron[b];
            neuron[b] = 1 / sqrt(layer1_size) * cos(neuron[b]);
            neuron[b + layer1_size] = 1 / sqrt(layer1_size) * sin(neuron[b + layer1_size]);
        }
        for (b = 0; b < layer1_size * 2; b++)
        {
            if (b < layer1_size * 2 - 1)
                fprintf(fp, "%lf ", neuron[b]);
            else
                fprintf(fp, "%lf\n", neuron[b]);
        }
    }
    fclose(fp);
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++)
    {
        if (!strcmp(str, argv[a]))
        {
            if (a == argc - 1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    FILE *fp;
    if (argc == 1)
    {
        printf("---User profile preserving network embedding---\n");
        printf("---The code and following instructions are built upon word2vec.c by Mikolov et al.---\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-graph <file>\n");
        printf("\t\tThe input <file> for network embedding\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting network embeddings\n");
        printf("\t-time <file>\n");
        printf("\t\tUse <file> to save running time\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of learned dimensions; default is 128\n");
        printf("\t-window <int>\n");
        printf("\t\tWindow size for collecting node context pairs; default is 10\n");
        printf("\t-walknum <int>\n");
        printf("\t\tThe number of random walks starting from per node; default is 40\n");
        printf("\t-walklen <int>\n");
        printf("\t\tThe length of random walks; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\t-samples <int>\n");
        printf("\t\tSet the number of training samples as <int> Million; default is 100\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-graph", argc, argv)) > 0) strcpy(graph_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(emb_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-time", argc, argv)) > 0) strcpy(time_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-walknum", argc, argv)) > 0) walk_num = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-walklen", argc, argv)) > 0) walk_length = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-samples", argc, argv)) >0) total_samples = atoi(argv[i + 1]);
    if (layer1_size % 2 != 0)
        printf("Warning: The embedding dimension must be even!\n");
    layer1_size = layer1_size / 2;
    total_samples = total_samples * 1000000;

    node_context_list = (struct node_context *)malloc(node_context_list_max_size * sizeof(struct node_context));
    node_context_hash = (long long *)malloc(node_context_hash_size * sizeof(long long));
    for (i = 0; i < node_context_hash_size; i++) node_context_hash[i] = -1;
    expTable = (double *)malloc((EXP_TABLE_SIZE + 1) * sizeof(double));
    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        expTable[i] = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    ReadGraph();
    start = clock();
    RandomWalk();
    free(node_context_hash);
    node_context_list = (struct node_context *)
        realloc(node_context_list, node_context_list_size * sizeof(struct node_context));
    InitNodeContextAliasTable();
    TrainModel();
    finish = clock();
    printf("Total time: %lf secs for learning node embeddings\n", (double)(finish - start) / CLOCKS_PER_SEC);
    printf("----------------------------------------------------\n");
    fp = fopen(time_file, "w");
    fprintf(fp, "Total time: %lf secs for learning node embeddings\n", (double)(finish - start) / CLOCKS_PER_SEC);
    fclose(fp);
    Output();
    return 0;
}
