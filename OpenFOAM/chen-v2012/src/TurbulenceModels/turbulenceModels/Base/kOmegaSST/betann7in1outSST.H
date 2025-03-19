const char* savedModelDir = "/home/chen/OpenFOAM/chen-v2012/src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/results_fullfield_ensemble_PH";
const char* tag = "serve";
const char* inputTensorName = "serving_default_cpp_norm_input";
const char* outputTensorName = "StatefulPartitionedCall";

// read the model
TF_Graph* Graph = TF_NewGraph();
TF_Status* Status = TF_NewStatus();
TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
TF_Buffer* RunOpts = NULL;
int ntags = 1;
TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, savedModelDir, &tag, ntags, Graph, NULL, Status);

// get input tensor
int NumInputs = 1;
TF_Output* Input = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumInputs));
TF_Output t0 = {TF_GraphOperationByName(Graph, inputTensorName), 0};
Input[0] = t0;
TF_Tensor** InputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumInputs));

// get output tensor
int NumOutputs = 1;
TF_Output* Output = static_cast<TF_Output*>(malloc(sizeof(TF_Output) * NumOutputs));
TF_Output t2 = {TF_GraphOperationByName(Graph, outputTensorName), 0};
if(t2.oper == NULL)
    printf("ERROR: Failed TF_GraphOperationByName %s\n", outputTensorName);
else	
printf("TF_GraphOperationByName %s is OK\n", outputTensorName);
Output[0] = t2;
TF_Tensor** OutputValues = static_cast<TF_Tensor**>(malloc(sizeof(TF_Tensor*)*NumOutputs));


//********* Set input data values
int ndims = 2; 
int ndata = sizeof(double)*nCells*nScalarInvariants; 
int64_t dims[] = {nCells,nScalarInvariants}; 
double* data = static_cast<double*>(malloc(ndata)); 
//double data[nCells*nScalarInvariants];
printf("set inputdata is ok! \n", outputTensorName);
for(int i=0;i< nCells; i++)
{
    data[7 * i+ 0] = theta1[i];
    data[7 * i+ 1] = theta2[i];
    data[7 * i+ 2] = theta3[i];
    data[7 * i+ 3] = theta4[i];
    data[7 * i+ 4] = theta5[i];
    data[7 * i+ 5] = theta6[i];
    data[7 * i+ 6] = theta7[i];
}

TF_Tensor* int_tensor1 = TF_NewTensor(TF_DOUBLE, dims, ndims, data, ndata, &NoOpDeallocator, 0);
InputValues[0] = int_tensor1;
printf("Set input data values is ok\n", outputTensorName);
//********* Run session
TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);
// free memory
TF_DeleteGraph(Graph);
TF_DeleteSession(Session, Status);
TF_DeleteSessionOptions(SessionOpts);
TF_DeleteStatus(Status);


//********* Read output data values
void* buff = TF_TensorData(OutputValues[0]);
double* nn = static_cast<double*>(buff);
printf("1 Read output data values is ok\n", outputTensorName);
for(int i=0;i<nCells;i++)
{
    betann[i] = 0.1*nn[i]+1;
}
printf("Read output data values is ok\n", outputTensorName);
// Free input tensor memory
for (int i = 0; i < NumInputs; i++) {
    if (InputValues[i] != nullptr) {
        TF_DeleteTensor(InputValues[i]);
    }
}
free(InputValues);

// Free output tensor memory
for (int i = 0; i < NumOutputs; i++) {
    if (OutputValues[i] != nullptr) {
        TF_DeleteTensor(OutputValues[i]);
    }
}
free(OutputValues);

free(data);
printf("Free input tensor memory is ok\n", outputTensorName);

