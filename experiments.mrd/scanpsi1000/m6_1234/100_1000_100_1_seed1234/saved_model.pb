̏
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-rc2-23-gb36436b0878�
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:d*
dtype0
q

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*
shared_name
Variable_1
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	d�*
dtype0
q

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_name
Variable_2
j
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:	�d*
dtype0
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:d*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
biholomorphic

layer1

layer2

layer3

layer4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
Y
w
	variables
regularization_losses
trainable_variables
	keras_api
Y
w
	variables
regularization_losses
trainable_variables
	keras_api
Y
w
	variables
regularization_losses
trainable_variables
	keras_api
Y
w
	variables
 regularization_losses
!trainable_variables
"	keras_api

0
1
2
3
 

0
1
2
3
�
#non_trainable_variables
	variables
$metrics
regularization_losses
trainable_variables

%layers
&layer_metrics
'layer_regularization_losses
 
 
 
 
�
(non_trainable_variables
	variables
)metrics
regularization_losses
trainable_variables

*layers
+layer_metrics
,layer_regularization_losses
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
-non_trainable_variables
	variables
.metrics
regularization_losses
trainable_variables

/layers
0layer_metrics
1layer_regularization_losses
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
2non_trainable_variables
	variables
3metrics
regularization_losses
trainable_variables

4layers
5layer_metrics
6layer_regularization_losses
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
7non_trainable_variables
	variables
8metrics
regularization_losses
trainable_variables

9layers
:layer_metrics
;layer_regularization_losses
CA
VARIABLE_VALUE
Variable_3#layer4/w/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
�
<non_trainable_variables
	variables
=metrics
 regularization_losses
!trainable_variables

>layers
?layer_metrics
@layer_regularization_losses
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable
Variable_1
Variable_2
Variable_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_368854
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_369006
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_369028ш
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_368792

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:���������d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_368854
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_3686792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
n
(__inference_dense_1_layer_call_fn_368942

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3687722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
1__inference_kahler_potential_layer_call_fn_368839
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_kahler_potential_layer_call_and_return_conditional_losses_3688252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
n
(__inference_dense_2_layer_call_fn_368957

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3687922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_368935

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_368772

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_368964

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_368811

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
C__inference_dense_2_layer_call_and_return_conditional_losses_368950

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:���������d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
(__inference_dense_3_layer_call_fn_368971

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3688112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
l
&__inference_dense_layer_call_fn_368927

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3687522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
L__inference_kahler_potential_layer_call_and_return_conditional_losses_368825
input_1
dense_368761
dense_1_368781
dense_2_368801
dense_3_368820
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_biholomorphic_layer_call_and_return_conditional_losses_3687362
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_368761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3687522
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_368781*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3687722!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_368801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3687922!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_368820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_3688112!
dense_3/StatefulPartitionedCallm
LogLog(dense_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Log�
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�9
e
I__inference_biholomorphic_layer_call_and_return_conditional_losses_368736

inputs
identityE
ConjConjinputs*'
_output_shapes
:���������2
Conj�
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:���������*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2
MatrixBandPart/num_upper�
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:���������2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:���������2
RealO
ImagImagReshape:output:0*'
_output_shapes
:���������2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2���������2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2���������2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape�
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack�
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1�
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2�
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice�
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices�
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1�
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack�
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1�
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2�
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2�
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack�
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1�
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2�
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2�
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axis�
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat�
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2���������2
boolean_mask/Reshape�
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
boolean_mask/Reshape_1/shape�
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:���������2
boolean_mask/Where�
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:���������*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis�
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:������������������2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm�
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:������������������2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�s
�
!__inference__wrapped_model_368679
input_19
5kahler_potential_dense_matmul_readvariableop_resource;
7kahler_potential_dense_1_matmul_readvariableop_resource;
7kahler_potential_dense_2_matmul_readvariableop_resource;
7kahler_potential_dense_3_matmul_readvariableop_resource
identity��
#kahler_potential/biholomorphic/ConjConjinput_1*'
_output_shapes
:���������2%
#kahler_potential/biholomorphic/Conj�
,kahler_potential/biholomorphic/einsum/EinsumEinsuminput_1,kahler_potential/biholomorphic/Conj:output:0*
N*
T0*+
_output_shapes
:���������*
equation
ai,aj->aij2.
,kahler_potential/biholomorphic/einsum/Einsum�
7kahler_potential/biholomorphic/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 29
7kahler_potential/biholomorphic/MatrixBandPart/num_lower�
7kahler_potential/biholomorphic/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������29
7kahler_potential/biholomorphic/MatrixBandPart/num_upper�
-kahler_potential/biholomorphic/MatrixBandPartMatrixBandPart5kahler_potential/biholomorphic/einsum/Einsum:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_lower:output:0@kahler_potential/biholomorphic/MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:���������2/
-kahler_potential/biholomorphic/MatrixBandPart�
,kahler_potential/biholomorphic/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2.
,kahler_potential/biholomorphic/Reshape/shape�
&kahler_potential/biholomorphic/ReshapeReshape4kahler_potential/biholomorphic/MatrixBandPart:band:05kahler_potential/biholomorphic/Reshape/shape:output:0*
T0*'
_output_shapes
:���������2(
&kahler_potential/biholomorphic/Reshape�
#kahler_potential/biholomorphic/RealReal/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:���������2%
#kahler_potential/biholomorphic/Real�
#kahler_potential/biholomorphic/ImagImag/kahler_potential/biholomorphic/Reshape:output:0*'
_output_shapes
:���������2%
#kahler_potential/biholomorphic/Imag�
*kahler_potential/biholomorphic/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2,
*kahler_potential/biholomorphic/concat/axis�
%kahler_potential/biholomorphic/concatConcatV2,kahler_potential/biholomorphic/Real:output:0,kahler_potential/biholomorphic/Imag:output:03kahler_potential/biholomorphic/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������22'
%kahler_potential/biholomorphic/concat�
-kahler_potential/biholomorphic/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2/
-kahler_potential/biholomorphic/transpose/perm�
(kahler_potential/biholomorphic/transpose	Transpose.kahler_potential/biholomorphic/concat:output:06kahler_potential/biholomorphic/transpose/perm:output:0*
T0*'
_output_shapes
:2���������2*
(kahler_potential/biholomorphic/transpose�
"kahler_potential/biholomorphic/AbsAbs,kahler_potential/biholomorphic/transpose:y:0*
T0*'
_output_shapes
:2���������2$
"kahler_potential/biholomorphic/Abs�
4kahler_potential/biholomorphic/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :26
4kahler_potential/biholomorphic/Sum/reduction_indices�
"kahler_potential/biholomorphic/SumSum&kahler_potential/biholomorphic/Abs:y:0=kahler_potential/biholomorphic/Sum/reduction_indices:output:0*
T0*
_output_shapes
:22$
"kahler_potential/biholomorphic/Sum�
%kahler_potential/biholomorphic/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2'
%kahler_potential/biholomorphic/Less/y�
#kahler_potential/biholomorphic/LessLess+kahler_potential/biholomorphic/Sum:output:0.kahler_potential/biholomorphic/Less/y:output:0*
T0*
_output_shapes
:22%
#kahler_potential/biholomorphic/Less�
)kahler_potential/biholomorphic/LogicalNot
LogicalNot'kahler_potential/biholomorphic/Less:z:0*
_output_shapes
:22+
)kahler_potential/biholomorphic/LogicalNot�
&kahler_potential/biholomorphic/SqueezeSqueeze-kahler_potential/biholomorphic/LogicalNot:y:0*
T0
*
_output_shapes
:22(
&kahler_potential/biholomorphic/Squeeze�
1kahler_potential/biholomorphic/boolean_mask/ShapeShape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:23
1kahler_potential/biholomorphic/boolean_mask/Shape�
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?kahler_potential/biholomorphic/boolean_mask/strided_slice/stack�
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1�
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2�
9kahler_potential/biholomorphic/boolean_mask/strided_sliceStridedSlice:kahler_potential/biholomorphic/boolean_mask/Shape:output:0Hkahler_potential/biholomorphic/boolean_mask/strided_slice/stack:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2;
9kahler_potential/biholomorphic/boolean_mask/strided_slice�
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indices�
0kahler_potential/biholomorphic/boolean_mask/ProdProdBkahler_potential/biholomorphic/boolean_mask/strided_slice:output:0Kkahler_potential/biholomorphic/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 22
0kahler_potential/biholomorphic/boolean_mask/Prod�
3kahler_potential/biholomorphic/boolean_mask/Shape_1Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_1�
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack�
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1�
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2�
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_1:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_1�
3kahler_potential/biholomorphic/boolean_mask/Shape_2Shape,kahler_potential/biholomorphic/transpose:y:0*
T0*
_output_shapes
:25
3kahler_potential/biholomorphic/boolean_mask/Shape_2�
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
Akahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack�
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1�
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ckahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2�
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2StridedSlice<kahler_potential/biholomorphic/boolean_mask/Shape_2:output:0Jkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_1:output:0Lkahler_potential/biholomorphic/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2=
;kahler_potential/biholomorphic/boolean_mask/strided_slice_2�
;kahler_potential/biholomorphic/boolean_mask/concat/values_1Pack9kahler_potential/biholomorphic/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2=
;kahler_potential/biholomorphic/boolean_mask/concat/values_1�
7kahler_potential/biholomorphic/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7kahler_potential/biholomorphic/boolean_mask/concat/axis�
2kahler_potential/biholomorphic/boolean_mask/concatConcatV2Dkahler_potential/biholomorphic/boolean_mask/strided_slice_1:output:0Dkahler_potential/biholomorphic/boolean_mask/concat/values_1:output:0Dkahler_potential/biholomorphic/boolean_mask/strided_slice_2:output:0@kahler_potential/biholomorphic/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2kahler_potential/biholomorphic/boolean_mask/concat�
3kahler_potential/biholomorphic/boolean_mask/ReshapeReshape,kahler_potential/biholomorphic/transpose:y:0;kahler_potential/biholomorphic/boolean_mask/concat:output:0*
T0*'
_output_shapes
:2���������25
3kahler_potential/biholomorphic/boolean_mask/Reshape�
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2=
;kahler_potential/biholomorphic/boolean_mask/Reshape_1/shape�
5kahler_potential/biholomorphic/boolean_mask/Reshape_1Reshape/kahler_potential/biholomorphic/Squeeze:output:0Dkahler_potential/biholomorphic/boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:227
5kahler_potential/biholomorphic/boolean_mask/Reshape_1�
1kahler_potential/biholomorphic/boolean_mask/WhereWhere>kahler_potential/biholomorphic/boolean_mask/Reshape_1:output:0*'
_output_shapes
:���������23
1kahler_potential/biholomorphic/boolean_mask/Where�
3kahler_potential/biholomorphic/boolean_mask/SqueezeSqueeze9kahler_potential/biholomorphic/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:���������*
squeeze_dims
25
3kahler_potential/biholomorphic/boolean_mask/Squeeze�
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9kahler_potential/biholomorphic/boolean_mask/GatherV2/axis�
4kahler_potential/biholomorphic/boolean_mask/GatherV2GatherV2<kahler_potential/biholomorphic/boolean_mask/Reshape:output:0<kahler_potential/biholomorphic/boolean_mask/Squeeze:output:0Bkahler_potential/biholomorphic/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:������������������26
4kahler_potential/biholomorphic/boolean_mask/GatherV2�
/kahler_potential/biholomorphic/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/kahler_potential/biholomorphic/transpose_1/perm�
*kahler_potential/biholomorphic/transpose_1	Transpose=kahler_potential/biholomorphic/boolean_mask/GatherV2:output:08kahler_potential/biholomorphic/transpose_1/perm:output:0*
T0*0
_output_shapes
:������������������2,
*kahler_potential/biholomorphic/transpose_1�
,kahler_potential/dense/MatMul/ReadVariableOpReadVariableOp5kahler_potential_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,kahler_potential/dense/MatMul/ReadVariableOp�
kahler_potential/dense/MatMulMatMul.kahler_potential/biholomorphic/transpose_1:y:04kahler_potential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
kahler_potential/dense/MatMul�
kahler_potential/dense/SquareSquare'kahler_potential/dense/MatMul:product:0*
T0*'
_output_shapes
:���������d2
kahler_potential/dense/Square�
.kahler_potential/dense_1/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype020
.kahler_potential/dense_1/MatMul/ReadVariableOp�
kahler_potential/dense_1/MatMulMatMul!kahler_potential/dense/Square:y:06kahler_potential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2!
kahler_potential/dense_1/MatMul�
kahler_potential/dense_1/SquareSquare)kahler_potential/dense_1/MatMul:product:0*
T0*(
_output_shapes
:����������2!
kahler_potential/dense_1/Square�
.kahler_potential/dense_2/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype020
.kahler_potential/dense_2/MatMul/ReadVariableOp�
kahler_potential/dense_2/MatMulMatMul#kahler_potential/dense_1/Square:y:06kahler_potential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2!
kahler_potential/dense_2/MatMul�
kahler_potential/dense_2/SquareSquare)kahler_potential/dense_2/MatMul:product:0*
T0*'
_output_shapes
:���������d2!
kahler_potential/dense_2/Square�
.kahler_potential/dense_3/MatMul/ReadVariableOpReadVariableOp7kahler_potential_dense_3_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.kahler_potential/dense_3/MatMul/ReadVariableOp�
kahler_potential/dense_3/MatMulMatMul#kahler_potential/dense_2/Square:y:06kahler_potential/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
kahler_potential/dense_3/MatMul�
kahler_potential/LogLog)kahler_potential/dense_3/MatMul:product:0*
T0*'
_output_shapes
:���������2
kahler_potential/Logl
IdentityIdentitykahler_potential/Log:y:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::::P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference__traced_save_369006
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a3b2b4ef1b6a49ec9a3ff8a4ae596885/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*A
_input_shapes0
.: :d:	d�:	�d:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d:%!

_output_shapes
:	d�:%!

_output_shapes
:	�d:$ 

_output_shapes

:d:

_output_shapes
: 
�
�
"__inference__traced_restore_369028
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
J
.__inference_biholomorphic_layer_call_fn_368912

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_biholomorphic_layer_call_and_return_conditional_losses_3687362
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_368920

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:���������d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_368752

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:���������d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�9
e
I__inference_biholomorphic_layer_call_and_return_conditional_losses_368907

inputs
identityE
ConjConjinputs*'
_output_shapes
:���������2
Conj�
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:���������*
equation
ai,aj->aij2
einsum/Einsumv
MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart/num_lower
MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
���������2
MatrixBandPart/num_upper�
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:���������2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:���������2
RealO
ImagImagReshape:output:0*'
_output_shapes
:���������2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis�
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������22
concatq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm
	transpose	Transposeconcat:output:0transpose/perm:output:0*
T0*'
_output_shapes
:2���������2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2���������2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:22
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2
Less/yX
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes
:22
LessL

LogicalNot
LogicalNotLess:z:0*
_output_shapes
:22

LogicalNotR
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes
:22	
Squeezee
boolean_mask/ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape�
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack�
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1�
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2�
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice�
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices�
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1�
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack�
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1�
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2�
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1i
boolean_mask/Shape_2Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2�
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack�
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1�
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2�
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2�
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axis�
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat�
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2���������2
boolean_mask/Reshape�
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
boolean_mask/Reshape_1/shape�
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:���������2
boolean_mask/Where�
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:���������*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis�
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:������������������2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm�
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:������������������2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�\
�
biholomorphic

layer1

layer2

layer3

layer4
	variables
regularization_losses
trainable_variables
		keras_api


signatures
A__call__
B_default_save_signature
*C&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
�
	variables
regularization_losses
trainable_variables
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
�
w
	variables
regularization_losses
trainable_variables
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
	variables
regularization_losses
trainable_variables
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
	variables
regularization_losses
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
	variables
 regularization_losses
!trainable_variables
"	keras_api
L__call__
*M&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
#non_trainable_variables
	variables
$metrics
regularization_losses
trainable_variables

%layers
&layer_metrics
'layer_regularization_losses
A__call__
B_default_save_signature
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
,
Nserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
(non_trainable_variables
	variables
)metrics
regularization_losses
trainable_variables

*layers
+layer_metrics
,layer_regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
-non_trainable_variables
	variables
.metrics
regularization_losses
trainable_variables

/layers
0layer_metrics
1layer_regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
:	d�2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
2non_trainable_variables
	variables
3metrics
regularization_losses
trainable_variables

4layers
5layer_metrics
6layer_regularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
:	�d2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
7non_trainable_variables
	variables
8metrics
regularization_losses
trainable_variables

9layers
:layer_metrics
;layer_regularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
<non_trainable_variables
	variables
=metrics
 regularization_losses
!trainable_variables

>layers
?layer_metrics
@layer_regularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�2�
1__inference_kahler_potential_layer_call_fn_368839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
!__inference__wrapped_model_368679�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
L__inference_kahler_potential_layer_call_and_return_conditional_losses_368825�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
.__inference_biholomorphic_layer_call_fn_368912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_biholomorphic_layer_call_and_return_conditional_losses_368907�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_368927�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_368920�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_368942�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_368935�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_368957�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_368950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_3_layer_call_fn_368971�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_3_layer_call_and_return_conditional_losses_368964�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
3B1
$__inference_signature_wrapper_368854input_1�
!__inference__wrapped_model_368679m0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
I__inference_biholomorphic_layer_call_and_return_conditional_losses_368907a/�,
%�"
 �
inputs���������
� ".�+
$�!
0������������������
� �
.__inference_biholomorphic_layer_call_fn_368912T/�,
%�"
 �
inputs���������
� "!��������������������
C__inference_dense_1_layer_call_and_return_conditional_losses_368935\/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� {
(__inference_dense_1_layer_call_fn_368942O/�,
%�"
 �
inputs���������d
� "������������
C__inference_dense_2_layer_call_and_return_conditional_losses_368950\0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� {
(__inference_dense_2_layer_call_fn_368957O0�-
&�#
!�
inputs����������
� "����������d�
C__inference_dense_3_layer_call_and_return_conditional_losses_368964[/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� z
(__inference_dense_3_layer_call_fn_368971N/�,
%�"
 �
inputs���������d
� "�����������
A__inference_dense_layer_call_and_return_conditional_losses_368920d8�5
.�+
)�&
inputs������������������
� "%�"
�
0���������d
� �
&__inference_dense_layer_call_fn_368927W8�5
.�+
)�&
inputs������������������
� "����������d�
L__inference_kahler_potential_layer_call_and_return_conditional_losses_368825_0�-
&�#
!�
input_1���������
� "%�"
�
0���������
� �
1__inference_kahler_potential_layer_call_fn_368839R0�-
&�#
!�
input_1���������
� "�����������
$__inference_signature_wrapper_368854x;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������