Á
ÍŁ
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
dtypetype
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18˘
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¤&*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	¤&*
dtype0

NoOpNoOp
ł	
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*î
valueäBá BÚ
Š
biholomorphic_k4

layer1

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
w
#	_self_saveable_object_factories

	variables
trainable_variables
regularization_losses
	keras_api
~
w
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 
 

0

0
 
­
	variables
trainable_variables
layer_regularization_losses
layer_metrics
metrics

layers
regularization_losses
non_trainable_variables
 
 
 
 
­

	variables
trainable_variables
layer_regularization_losses
layer_metrics
metrics

layers
regularization_losses
non_trainable_variables
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
­
	variables
trainable_variables
layer_regularization_losses
layer_metrics
 metrics

!layers
regularization_losses
"non_trainable_variables
 
 
 

0
1
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
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_108853
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
˝
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_108879
¤
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable*
Tin
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_108892ň
é

z
"__inference__traced_restore_108892
file_prefix
assignvariableop_variable

identity_2˘AssignVariableOpÉ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesľ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes

::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp9
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp{

Identity_1Identityfile_prefix^AssignVariableOp^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_1m

Identity_2IdentityIdentity_1:output:0^AssignVariableOp*
T0*
_output_shapes
: 2

Identity_2"!

identity_2Identity_2:output:0*
_input_shapes
: :2$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ł
K
/__inference_biholomorphic_k4_layer_call_fn_1911

inputs
identityÔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_18742
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š

I__inference_width_one_dense_layer_call_and_return_conditional_losses_2217

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¤&*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

ł
G__forward_width_one_dense_layer_call_and_return_conditional_losses_3514
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¤&*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:*u
backward_function_name[Y__inference___backward_width_one_dense_layer_call_and_return_conditional_losses_3498_3515:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Š

I__inference_width_one_dense_layer_call_and_return_conditional_losses_1757

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¤&*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ô
y
2__inference_outer_product_nn_k4_layer_call_fn_1906
input_1
unknown
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_outer_product_nn_k4_layer_call_and_return_conditional_losses_18822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
´1
¨
K__forward_outer_product_nn_k4_layer_call_and_return_conditional_losses_3735
input_1
width_one_dense_10553061
identity+
'width_one_dense_statefulpartitionedcall-
)width_one_dense_statefulpartitionedcall_0-
)width_one_dense_statefulpartitionedcall_1$
 biholomorphic_k4_partitionedcall&
"biholomorphic_k4_partitionedcall_0&
"biholomorphic_k4_partitionedcall_1	&
"biholomorphic_k4_partitionedcall_2&
"biholomorphic_k4_partitionedcall_3&
"biholomorphic_k4_partitionedcall_4&
"biholomorphic_k4_partitionedcall_5&
"biholomorphic_k4_partitionedcall_6&
"biholomorphic_k4_partitionedcall_7&
"biholomorphic_k4_partitionedcall_8	&
"biholomorphic_k4_partitionedcall_9	'
#biholomorphic_k4_partitionedcall_10'
#biholomorphic_k4_partitionedcall_11'
#biholomorphic_k4_partitionedcall_12'
#biholomorphic_k4_partitionedcall_13'
#biholomorphic_k4_partitionedcall_14'
#biholomorphic_k4_partitionedcall_15	'
#biholomorphic_k4_partitionedcall_16'
#biholomorphic_k4_partitionedcall_17'
#biholomorphic_k4_partitionedcall_18'
#biholomorphic_k4_partitionedcall_19	'
#biholomorphic_k4_partitionedcall_20	'
#biholomorphic_k4_partitionedcall_21'
#biholomorphic_k4_partitionedcall_22	'
#biholomorphic_k4_partitionedcall_23	'
#biholomorphic_k4_partitionedcall_24'
#biholomorphic_k4_partitionedcall_25	'
#biholomorphic_k4_partitionedcall_26	'
#biholomorphic_k4_partitionedcall_27'
#biholomorphic_k4_partitionedcall_28'
#biholomorphic_k4_partitionedcall_29˘'width_one_dense/StatefulPartitionedCall
 biholomorphic_k4/PartitionedCallPartitionedCallinput_1*
Tin
2*,
Tout$
"2 										*
_collective_manager_ids
 *ˇ
_output_shapes¤
Ą:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ČL˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ČL˙˙˙˙˙˙˙˙˙:: :˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙FF: : :˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ń˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ń˙˙˙˙˙˙˙˙˙::˙˙˙˙˙˙˙˙˙: : :: : :: : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__forward_biholomorphic_k4_layer_call_and_return_conditional_losses_36992"
 biholomorphic_k4/PartitionedCallę
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k4/PartitionedCall:output:0width_one_dense_10553061*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::˙˙˙˙˙˙˙˙˙:	¤&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__forward_width_one_dense_layer_call_and_return_conditional_losses_35142)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"M
 biholomorphic_k4_partitionedcall)biholomorphic_k4/PartitionedCall:output:1"O
"biholomorphic_k4_partitionedcall_0)biholomorphic_k4/PartitionedCall:output:2"O
"biholomorphic_k4_partitionedcall_1)biholomorphic_k4/PartitionedCall:output:3"Q
#biholomorphic_k4_partitionedcall_10*biholomorphic_k4/PartitionedCall:output:12"Q
#biholomorphic_k4_partitionedcall_11*biholomorphic_k4/PartitionedCall:output:13"Q
#biholomorphic_k4_partitionedcall_12*biholomorphic_k4/PartitionedCall:output:14"Q
#biholomorphic_k4_partitionedcall_13*biholomorphic_k4/PartitionedCall:output:15"Q
#biholomorphic_k4_partitionedcall_14*biholomorphic_k4/PartitionedCall:output:16"Q
#biholomorphic_k4_partitionedcall_15*biholomorphic_k4/PartitionedCall:output:17"Q
#biholomorphic_k4_partitionedcall_16*biholomorphic_k4/PartitionedCall:output:18"Q
#biholomorphic_k4_partitionedcall_17*biholomorphic_k4/PartitionedCall:output:19"Q
#biholomorphic_k4_partitionedcall_18*biholomorphic_k4/PartitionedCall:output:20"Q
#biholomorphic_k4_partitionedcall_19*biholomorphic_k4/PartitionedCall:output:21"O
"biholomorphic_k4_partitionedcall_2)biholomorphic_k4/PartitionedCall:output:4"Q
#biholomorphic_k4_partitionedcall_20*biholomorphic_k4/PartitionedCall:output:22"Q
#biholomorphic_k4_partitionedcall_21*biholomorphic_k4/PartitionedCall:output:23"Q
#biholomorphic_k4_partitionedcall_22*biholomorphic_k4/PartitionedCall:output:24"Q
#biholomorphic_k4_partitionedcall_23*biholomorphic_k4/PartitionedCall:output:25"Q
#biholomorphic_k4_partitionedcall_24*biholomorphic_k4/PartitionedCall:output:26"Q
#biholomorphic_k4_partitionedcall_25*biholomorphic_k4/PartitionedCall:output:27"Q
#biholomorphic_k4_partitionedcall_26*biholomorphic_k4/PartitionedCall:output:28"Q
#biholomorphic_k4_partitionedcall_27*biholomorphic_k4/PartitionedCall:output:29"Q
#biholomorphic_k4_partitionedcall_28*biholomorphic_k4/PartitionedCall:output:30"Q
#biholomorphic_k4_partitionedcall_29*biholomorphic_k4/PartitionedCall:output:31"O
"biholomorphic_k4_partitionedcall_3)biholomorphic_k4/PartitionedCall:output:5"O
"biholomorphic_k4_partitionedcall_4)biholomorphic_k4/PartitionedCall:output:6"O
"biholomorphic_k4_partitionedcall_5)biholomorphic_k4/PartitionedCall:output:7"O
"biholomorphic_k4_partitionedcall_6)biholomorphic_k4/PartitionedCall:output:8"O
"biholomorphic_k4_partitionedcall_7)biholomorphic_k4/PartitionedCall:output:9"P
"biholomorphic_k4_partitionedcall_8*biholomorphic_k4/PartitionedCall:output:10"P
"biholomorphic_k4_partitionedcall_9*biholomorphic_k4/PartitionedCall:output:11"
identityIdentity:output:0"[
'width_one_dense_statefulpartitionedcall0width_one_dense/StatefulPartitionedCall:output:0"]
)width_one_dense_statefulpartitionedcall_00width_one_dense/StatefulPartitionedCall:output:1"]
)width_one_dense_statefulpartitionedcall_10width_one_dense/StatefulPartitionedCall:output:2**
_input_shapes
:˙˙˙˙˙˙˙˙˙:*y
backward_function_name_]__inference___backward_outer_product_nn_k4_layer_call_and_return_conditional_losses_3456_37362R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ű
t
.__inference_width_one_dense_layer_call_fn_1763

inputs
unknown
identity˘StatefulPartitionedCallď
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_width_one_dense_layer_call_and_return_conditional_losses_17572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ö
f
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_1874

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
aj,ai->aji2
einsum/Einsum
einsum/Einsum_1Einsuminputsinputs*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
al,ak->alk2
einsum/Einsum_1Ă
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0einsum/Einsum:output:0*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
equationalk,aji->aijkl2
einsum/Einsum_2v
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
˙˙˙˙˙˙˙˙˙2
MatrixBandPart/num_upperĐ
MatrixBandPartMatrixBandParteinsum/Einsum_2:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	TransposeMatrixBandPart:band:0transpose/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
	transposez
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1/num_upperÍ
MatrixBandPart_1MatrixBandParttranspose:y:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm
transpose_1	TransposeMatrixBandPart_1:band:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
transpose_1z
MatrixBandPart_2/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_2/num_lower
MatrixBandPart_2/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2/num_upperĎ
MatrixBandPart_2MatrixBandParttranspose_1:y:0#MatrixBandPart_2/num_lower:output:0#MatrixBandPart_2/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙q  2
Reshape/shape
ReshapeReshapeMatrixBandPart_2:band:0Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ń2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
transpose_2S
Abs
ComplexAbstranspose_2:y:0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices`
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:ń2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yY
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes	
:ń2
LessM

LogicalNot
LogicalNotLess:z:0*
_output_shapes	
:ń2

LogicalNotS
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes	
:ń2	
Squeezeg
boolean_mask/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices˘
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2¸
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2¸
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
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
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_2:y:0boolean_mask/concat:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ń2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisű
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transposeboolean_mask/GatherV2:output:0transpose_3/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙F   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_3:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
Conj¨
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_3/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_3/num_lower
MatrixBandPart_3/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_3/num_upperĐ
MatrixBandPart_3MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_3/num_lower:output:0#MatrixBandPart_3/num_upper:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF2
MatrixBandPart_3s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙$  2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_3:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ČL2
concatu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposeconcat:output:0transpose_4/perm:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
transpose_4Y
Abs_1Abstranspose_4:y:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:ČL2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:ČL2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:ČL2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:ČL2
	Squeeze_1k
boolean_mask_1/ShapeShapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2¨
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŞ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodo
boolean_mask_1/Shape_1Shapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1o
boolean_mask_1/Shape_2Shapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concat
boolean_mask_1/ReshapeReshapetranspose_4:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
boolean_mask_1/Reshape_1/shape˘
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ČL2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask_1/GatherV2u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transpose boolean_mask_1/GatherV2:output:0transpose_5/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_5l
IdentityIdentitytranspose_5:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
 

__inference__traced_save_108879
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f813e1b03b494bec8b24535d04715901/part2	
Const_1
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameĂ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesŕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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

identity_1Identity_1:output:0*"
_input_shapes
: :	¤&: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	¤&:

_output_shapes
: 

ł
G__forward_width_one_dense_layer_call_and_return_conditional_losses_2500
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	¤&*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:*u
backward_function_name[Y__inference___backward_width_one_dense_layer_call_and_return_conditional_losses_2490_2501:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
Ą
H__forward_biholomorphic_k4_layer_call_and_return_conditional_losses_3699
inputs_0
identity
transpose_5_perm
boolean_mask_1_reshape
boolean_mask_1_squeeze	
transpose_4
transpose_4_perm
concat_axis
real
imag
matrixbandpart_3
matrixbandpart_3_num_lower	
matrixbandpart_3_num_upper	
	reshape_1
conj
transpose_3
transpose_3_perm
boolean_mask_reshape
boolean_mask_squeeze	
transpose_2
transpose_2_perm
matrixbandpart_2
matrixbandpart_2_num_lower	
matrixbandpart_2_num_upper	
transpose_1_perm
matrixbandpart_1_num_lower	
matrixbandpart_1_num_upper	
transpose_perm
matrixbandpart_num_lower	
matrixbandpart_num_upper	
einsum_einsum_1
einsum_einsum

inputs
einsum/EinsumEinsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
aj,ai->aji2
einsum/Einsum
einsum/Einsum_1Einsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
al,ak->alk2
einsum/Einsum_1Ă
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0einsum/Einsum:output:0*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
equationalk,aji->aijkl2
einsum/Einsum_2v
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
˙˙˙˙˙˙˙˙˙2
MatrixBandPart/num_upperĐ
MatrixBandPartMatrixBandParteinsum/Einsum_2:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	TransposeMatrixBandPart:band:0transpose/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
	transposez
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1/num_upperÍ
MatrixBandPart_1MatrixBandParttranspose:y:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm
transpose_1	TransposeMatrixBandPart_1:band:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
transpose_1z
MatrixBandPart_2/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_2/num_lower
MatrixBandPart_2/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2/num_upperĎ
MatrixBandPart_2MatrixBandParttranspose_1:y:0#MatrixBandPart_2/num_lower:output:0#MatrixBandPart_2/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙q  2
Reshape/shape
ReshapeReshapeMatrixBandPart_2:band:0Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ń2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	TransposeReshape:output:0transpose_2/perm:output:0*
T02
transpose_2U
Abs
ComplexAbstranspose_2_0:y:0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices`
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:ń2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yY
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes	
:ń2
LessM

LogicalNot
LogicalNotLess:z:0*
_output_shapes	
:ń2

LogicalNotS
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes	
:ń2	
Squeezei
boolean_mask/ShapeShapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices˘
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodm
boolean_mask/Shape_1Shapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2¸
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1m
boolean_mask/Shape_2Shapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2¸
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
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
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_2_0:y:0boolean_mask/concat:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ń2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisű
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/permm
transpose_3_0	Transposeboolean_mask/GatherV2:output:0transpose_3/perm:output:0*
T02
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙F   2
Reshape_1/shape
	Reshape_1Reshapetranspose_3_0:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
Conj¨
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_3/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_3/num_lower
MatrixBandPart_3/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_3/num_upperĐ
MatrixBandPart_3MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_3/num_lower:output:0#MatrixBandPart_3/num_upper:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF2
MatrixBandPart_3s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙$  2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_3:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ČL2
concatu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm^
transpose_4_0	Transposeconcat:output:0transpose_4/perm:output:0*
T02
transpose_4[
Abs_1Abstranspose_4_0:y:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:ČL2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:ČL2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:ČL2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:ČL2
	Squeeze_1m
boolean_mask_1/ShapeShapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2¨
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŞ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodq
boolean_mask_1/Shape_1Shapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1q
boolean_mask_1/Shape_2Shapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concatĄ
boolean_mask_1/ReshapeReshapetranspose_4_0:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
boolean_mask_1/Reshape_1/shape˘
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ČL2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask_1/GatherV2u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transpose boolean_mask_1/GatherV2:output:0transpose_5/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_5l
IdentityIdentitytranspose_5:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"9
boolean_mask_1_reshapeboolean_mask_1/Reshape:output:0"9
boolean_mask_1_squeezeboolean_mask_1/Squeeze:output:0"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"'
einsum_einsumeinsum/Einsum:output:0"+
einsum_einsum_1einsum/Einsum_1:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"A
matrixbandpart_1_num_lower#MatrixBandPart_1/num_lower:output:0"A
matrixbandpart_1_num_upper#MatrixBandPart_1/num_upper:output:0"+
matrixbandpart_2MatrixBandPart_2:band:0"A
matrixbandpart_2_num_lower#MatrixBandPart_2/num_lower:output:0"A
matrixbandpart_2_num_upper#MatrixBandPart_2/num_upper:output:0"+
matrixbandpart_3MatrixBandPart_3:band:0"A
matrixbandpart_3_num_lower#MatrixBandPart_3/num_lower:output:0"A
matrixbandpart_3_num_upper#MatrixBandPart_3/num_upper:output:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	reshape_1Reshape_1:output:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙*v
backward_function_name\Z__inference___backward_biholomorphic_k4_layer_call_and_return_conditional_losses_3520_3700:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ĺ
Ą
H__forward_biholomorphic_k4_layer_call_and_return_conditional_losses_2647
inputs_0
identity
transpose_5_perm
boolean_mask_1_reshape
boolean_mask_1_squeeze	
transpose_4
transpose_4_perm
concat_axis
real
imag
matrixbandpart_3
matrixbandpart_3_num_lower	
matrixbandpart_3_num_upper	
	reshape_1
conj
transpose_3
transpose_3_perm
boolean_mask_reshape
boolean_mask_squeeze	
transpose_2
transpose_2_perm
matrixbandpart_2
matrixbandpart_2_num_lower	
matrixbandpart_2_num_upper	
transpose_1_perm
matrixbandpart_1_num_lower	
matrixbandpart_1_num_upper	
transpose_perm
matrixbandpart_num_lower	
matrixbandpart_num_upper	
einsum_einsum_1
einsum_einsum

inputs
einsum/EinsumEinsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
aj,ai->aji2
einsum/Einsum
einsum/Einsum_1Einsuminputs_0inputs_0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
al,ak->alk2
einsum/Einsum_1Ă
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0einsum/Einsum:output:0*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
equationalk,aji->aijkl2
einsum/Einsum_2v
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
˙˙˙˙˙˙˙˙˙2
MatrixBandPart/num_upperĐ
MatrixBandPartMatrixBandParteinsum/Einsum_2:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	TransposeMatrixBandPart:band:0transpose/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
	transposez
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1/num_upperÍ
MatrixBandPart_1MatrixBandParttranspose:y:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm
transpose_1	TransposeMatrixBandPart_1:band:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
transpose_1z
MatrixBandPart_2/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_2/num_lower
MatrixBandPart_2/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2/num_upperĎ
MatrixBandPart_2MatrixBandParttranspose_1:y:0#MatrixBandPart_2/num_lower:output:0#MatrixBandPart_2/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙q  2
Reshape/shape
ReshapeReshapeMatrixBandPart_2:band:0Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ń2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm_
transpose_2_0	TransposeReshape:output:0transpose_2/perm:output:0*
T02
transpose_2U
Abs
ComplexAbstranspose_2_0:y:0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices`
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:ń2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yY
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes	
:ń2
LessM

LogicalNot
LogicalNotLess:z:0*
_output_shapes	
:ń2

LogicalNotS
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes	
:ń2	
Squeezei
boolean_mask/ShapeShapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices˘
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodm
boolean_mask/Shape_1Shapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2¸
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1m
boolean_mask/Shape_2Shapetranspose_2_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2¸
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
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
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_2_0:y:0boolean_mask/concat:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ń2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisű
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/permm
transpose_3_0	Transposeboolean_mask/GatherV2:output:0transpose_3/perm:output:0*
T02
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙F   2
Reshape_1/shape
	Reshape_1Reshapetranspose_3_0:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
Conj¨
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_3/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_3/num_lower
MatrixBandPart_3/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_3/num_upperĐ
MatrixBandPart_3MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_3/num_lower:output:0#MatrixBandPart_3/num_upper:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF2
MatrixBandPart_3s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙$  2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_3:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ČL2
concatu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm^
transpose_4_0	Transposeconcat:output:0transpose_4/perm:output:0*
T02
transpose_4[
Abs_1Abstranspose_4_0:y:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:ČL2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:ČL2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:ČL2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:ČL2
	Squeeze_1m
boolean_mask_1/ShapeShapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2¨
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŞ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodq
boolean_mask_1/Shape_1Shapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1q
boolean_mask_1/Shape_2Shapetranspose_4_0:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concatĄ
boolean_mask_1/ReshapeReshapetranspose_4_0:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
boolean_mask_1/Reshape_1/shape˘
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ČL2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask_1/GatherV2u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transpose boolean_mask_1/GatherV2:output:0transpose_5/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_5l
IdentityIdentitytranspose_5:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"9
boolean_mask_1_reshapeboolean_mask_1/Reshape:output:0"9
boolean_mask_1_squeezeboolean_mask_1/Squeeze:output:0"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"'
einsum_einsumeinsum/Einsum:output:0"+
einsum_einsum_1einsum/Einsum_1:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"A
matrixbandpart_1_num_lower#MatrixBandPart_1/num_lower:output:0"A
matrixbandpart_1_num_upper#MatrixBandPart_1/num_upper:output:0"+
matrixbandpart_2MatrixBandPart_2:band:0"A
matrixbandpart_2_num_lower#MatrixBandPart_2/num_lower:output:0"A
matrixbandpart_2_num_upper#MatrixBandPart_2/num_upper:output:0"+
matrixbandpart_3MatrixBandPart_3:band:0"A
matrixbandpart_3_num_lower#MatrixBandPart_3/num_lower:output:0"A
matrixbandpart_3_num_upper#MatrixBandPart_3/num_upper:output:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	reshape_1Reshape_1:output:0"-
transpose_1_permtranspose_1/perm:output:0" 
transpose_2transpose_2_0:y:0"-
transpose_2_permtranspose_2/perm:output:0" 
transpose_3transpose_3_0:y:0"-
transpose_3_permtranspose_3/perm:output:0" 
transpose_4transpose_4_0:y:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙*v
backward_function_name\Z__inference___backward_biholomorphic_k4_layer_call_and_return_conditional_losses_2508_2648:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
´1
¨
K__forward_outer_product_nn_k4_layer_call_and_return_conditional_losses_2714
input_1
width_one_dense_10553061
identity+
'width_one_dense_statefulpartitionedcall-
)width_one_dense_statefulpartitionedcall_0-
)width_one_dense_statefulpartitionedcall_1$
 biholomorphic_k4_partitionedcall&
"biholomorphic_k4_partitionedcall_0&
"biholomorphic_k4_partitionedcall_1	&
"biholomorphic_k4_partitionedcall_2&
"biholomorphic_k4_partitionedcall_3&
"biholomorphic_k4_partitionedcall_4&
"biholomorphic_k4_partitionedcall_5&
"biholomorphic_k4_partitionedcall_6&
"biholomorphic_k4_partitionedcall_7&
"biholomorphic_k4_partitionedcall_8	&
"biholomorphic_k4_partitionedcall_9	'
#biholomorphic_k4_partitionedcall_10'
#biholomorphic_k4_partitionedcall_11'
#biholomorphic_k4_partitionedcall_12'
#biholomorphic_k4_partitionedcall_13'
#biholomorphic_k4_partitionedcall_14'
#biholomorphic_k4_partitionedcall_15	'
#biholomorphic_k4_partitionedcall_16'
#biholomorphic_k4_partitionedcall_17'
#biholomorphic_k4_partitionedcall_18'
#biholomorphic_k4_partitionedcall_19	'
#biholomorphic_k4_partitionedcall_20	'
#biholomorphic_k4_partitionedcall_21'
#biholomorphic_k4_partitionedcall_22	'
#biholomorphic_k4_partitionedcall_23	'
#biholomorphic_k4_partitionedcall_24'
#biholomorphic_k4_partitionedcall_25	'
#biholomorphic_k4_partitionedcall_26	'
#biholomorphic_k4_partitionedcall_27'
#biholomorphic_k4_partitionedcall_28'
#biholomorphic_k4_partitionedcall_29˘'width_one_dense/StatefulPartitionedCall
 biholomorphic_k4/PartitionedCallPartitionedCallinput_1*
Tin
2*,
Tout$
"2 										*
_collective_manager_ids
 *ˇ
_output_shapes¤
Ą:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ČL˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ČL˙˙˙˙˙˙˙˙˙:: :˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙FF: : :˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ń˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ń˙˙˙˙˙˙˙˙˙::˙˙˙˙˙˙˙˙˙: : :: : :: : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__forward_biholomorphic_k4_layer_call_and_return_conditional_losses_26472"
 biholomorphic_k4/PartitionedCallę
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k4/PartitionedCall:output:0width_one_dense_10553061*
Tin
2*
Tout
2*
_collective_manager_ids
 *N
_output_shapes<
::˙˙˙˙˙˙˙˙˙:	¤&:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__forward_width_one_dense_layer_call_and_return_conditional_losses_25002)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"M
 biholomorphic_k4_partitionedcall)biholomorphic_k4/PartitionedCall:output:1"O
"biholomorphic_k4_partitionedcall_0)biholomorphic_k4/PartitionedCall:output:2"O
"biholomorphic_k4_partitionedcall_1)biholomorphic_k4/PartitionedCall:output:3"Q
#biholomorphic_k4_partitionedcall_10*biholomorphic_k4/PartitionedCall:output:12"Q
#biholomorphic_k4_partitionedcall_11*biholomorphic_k4/PartitionedCall:output:13"Q
#biholomorphic_k4_partitionedcall_12*biholomorphic_k4/PartitionedCall:output:14"Q
#biholomorphic_k4_partitionedcall_13*biholomorphic_k4/PartitionedCall:output:15"Q
#biholomorphic_k4_partitionedcall_14*biholomorphic_k4/PartitionedCall:output:16"Q
#biholomorphic_k4_partitionedcall_15*biholomorphic_k4/PartitionedCall:output:17"Q
#biholomorphic_k4_partitionedcall_16*biholomorphic_k4/PartitionedCall:output:18"Q
#biholomorphic_k4_partitionedcall_17*biholomorphic_k4/PartitionedCall:output:19"Q
#biholomorphic_k4_partitionedcall_18*biholomorphic_k4/PartitionedCall:output:20"Q
#biholomorphic_k4_partitionedcall_19*biholomorphic_k4/PartitionedCall:output:21"O
"biholomorphic_k4_partitionedcall_2)biholomorphic_k4/PartitionedCall:output:4"Q
#biholomorphic_k4_partitionedcall_20*biholomorphic_k4/PartitionedCall:output:22"Q
#biholomorphic_k4_partitionedcall_21*biholomorphic_k4/PartitionedCall:output:23"Q
#biholomorphic_k4_partitionedcall_22*biholomorphic_k4/PartitionedCall:output:24"Q
#biholomorphic_k4_partitionedcall_23*biholomorphic_k4/PartitionedCall:output:25"Q
#biholomorphic_k4_partitionedcall_24*biholomorphic_k4/PartitionedCall:output:26"Q
#biholomorphic_k4_partitionedcall_25*biholomorphic_k4/PartitionedCall:output:27"Q
#biholomorphic_k4_partitionedcall_26*biholomorphic_k4/PartitionedCall:output:28"Q
#biholomorphic_k4_partitionedcall_27*biholomorphic_k4/PartitionedCall:output:29"Q
#biholomorphic_k4_partitionedcall_28*biholomorphic_k4/PartitionedCall:output:30"Q
#biholomorphic_k4_partitionedcall_29*biholomorphic_k4/PartitionedCall:output:31"O
"biholomorphic_k4_partitionedcall_3)biholomorphic_k4/PartitionedCall:output:5"O
"biholomorphic_k4_partitionedcall_4)biholomorphic_k4/PartitionedCall:output:6"O
"biholomorphic_k4_partitionedcall_5)biholomorphic_k4/PartitionedCall:output:7"O
"biholomorphic_k4_partitionedcall_6)biholomorphic_k4/PartitionedCall:output:8"O
"biholomorphic_k4_partitionedcall_7)biholomorphic_k4/PartitionedCall:output:9"P
"biholomorphic_k4_partitionedcall_8*biholomorphic_k4/PartitionedCall:output:10"P
"biholomorphic_k4_partitionedcall_9*biholomorphic_k4/PartitionedCall:output:11"
identityIdentity:output:0"[
'width_one_dense_statefulpartitionedcall0width_one_dense/StatefulPartitionedCall:output:0"]
)width_one_dense_statefulpartitionedcall_00width_one_dense/StatefulPartitionedCall:output:1"]
)width_one_dense_statefulpartitionedcall_10width_one_dense/StatefulPartitionedCall:output:2**
_input_shapes
:˙˙˙˙˙˙˙˙˙:*y
backward_function_name_]__inference___backward_outer_product_nn_k4_layer_call_and_return_conditional_losses_2483_27152R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
đ
ľ
M__inference_outer_product_nn_k4_layer_call_and_return_conditional_losses_1882
input_1
width_one_dense_10553061
identity˘'width_one_dense/StatefulPartitionedCall˙
 biholomorphic_k4/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *ˇ
_output_shapes¤
Ą:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ČL˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ČL˙˙˙˙˙˙˙˙˙:: :˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙¤&:˙˙˙˙˙˙˙˙˙FF: : :˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙F:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙::ń˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:ń˙˙˙˙˙˙˙˙˙::˙˙˙˙˙˙˙˙˙: : :: : :: : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_18742"
 biholomorphic_k4/PartitionedCallĂ
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall)biholomorphic_k4/PartitionedCall:output:0width_one_dense_10553061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_width_one_dense_layer_call_and_return_conditional_losses_17572)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Log
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:2R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1

k
$__inference_signature_wrapper_108853
input_1
unknown
identity˘StatefulPartitionedCallČ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1088442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Đ

!__inference__wrapped_model_108844
input_1
outer_product_nn_k4_108840
identity˘+outer_product_nn_k4/StatefulPartitionedCall
+outer_product_nn_k4/StatefulPartitionedCallStatefulPartitionedCallinput_1outer_product_nn_k4_108840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_restored_function_body_24722-
+outer_product_nn_k4/StatefulPartitionedCallś
IdentityIdentity4outer_product_nn_k4/StatefulPartitionedCall:output:0,^outer_product_nn_k4/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:2Z
+outer_product_nn_k4/StatefulPartitionedCall+outer_product_nn_k4/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ş
n
'__inference_restored_function_body_2472
input_1
unknown
identity˘StatefulPartitionedCallŐ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_outer_product_nn_k4_layer_call_and_return_conditional_losses_18822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0**
_input_shapes
:˙˙˙˙˙˙˙˙˙:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1
Ö
f
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_2328

inputs
identity
einsum/EinsumEinsuminputsinputs*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
aj,ai->aji2
einsum/Einsum
einsum/Einsum_1Einsuminputsinputs*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equation
al,ak->alk2
einsum/Einsum_1Ă
einsum/Einsum_2Einsumeinsum/Einsum_1:output:0einsum/Einsum:output:0*
N*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙*
equationalk,aji->aijkl2
einsum/Einsum_2v
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
˙˙˙˙˙˙˙˙˙2
MatrixBandPart/num_upperĐ
MatrixBandPartMatrixBandParteinsum/Einsum_2:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	TransposeMatrixBandPart:band:0transpose/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
	transposez
MatrixBandPart_1/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_1/num_lower
MatrixBandPart_1/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1/num_upperÍ
MatrixBandPart_1MatrixBandParttranspose:y:0#MatrixBandPart_1/num_lower:output:0#MatrixBandPart_1/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_1
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm
transpose_1	TransposeMatrixBandPart_1:band:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
transpose_1z
MatrixBandPart_2/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_2/num_lower
MatrixBandPart_2/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2/num_upperĎ
MatrixBandPart_2MatrixBandParttranspose_1:y:0#MatrixBandPart_2/num_lower:output:0#MatrixBandPart_2/num_upper:output:0*
T0*3
_output_shapes!
:˙˙˙˙˙˙˙˙˙2
MatrixBandPart_2o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙q  2
Reshape/shape
ReshapeReshapeMatrixBandPart_2:band:0Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ń2	
Reshapeu
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	TransposeReshape:output:0transpose_2/perm:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
transpose_2S
Abs
ComplexAbstranspose_2:y:0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
Absp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices`
SumSumAbs:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes	
:ń2
SumU
Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
Less/yY
LessLessSum:output:0Less/y:output:0*
T0*
_output_shapes	
:ń2
LessM

LogicalNot
LogicalNotLess:z:0*
_output_shapes	
:ń2

LogicalNotS
SqueezeSqueezeLogicalNot:y:0*
T0
*
_output_shapes	
:ń2	
Squeezeg
boolean_mask/ShapeShapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices˘
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2¸
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_2:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_2
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2¸
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2
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
boolean_mask/concat/axisü
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat
boolean_mask/ReshapeReshapetranspose_2:y:0boolean_mask/concat:output:0*
T0*(
_output_shapes
:ń˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2
boolean_mask/Reshape_1/shape
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ń2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask/Where
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisű
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask/GatherV2u
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_3/perm
transpose_3	Transposeboolean_mask/GatherV2:output:0transpose_3/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_3s
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙F   2
Reshape_1/shape~
	Reshape_1Reshapetranspose_3:y:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
	Reshape_1Q
ConjConjReshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙F2
Conj¨
einsum_1/EinsumEinsumReshape_1:output:0Conj:output:0*
N*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF*
equation
ai,aj->aij2
einsum_1/Einsumz
MatrixBandPart_3/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
MatrixBandPart_3/num_lower
MatrixBandPart_3/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
˙˙˙˙˙˙˙˙˙2
MatrixBandPart_3/num_upperĐ
MatrixBandPart_3MatrixBandParteinsum_1/Einsum:output:0#MatrixBandPart_3/num_lower:output:0#MatrixBandPart_3/num_upper:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙FF2
MatrixBandPart_3s
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙$  2
Reshape_2/shape
	Reshape_2ReshapeMatrixBandPart_3:band:0Reshape_2/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
	Reshape_2R
RealRealReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
RealR
ImagImagReshape_2:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤&2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ČL2
concatu
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_4/perm
transpose_4	Transposeconcat:output:0transpose_4/perm:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
transpose_4Y
Abs_1Abstranspose_4:y:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
Abs_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesh
Sum_1Sum	Abs_1:y:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:ČL2
Sum_1Y
Less_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2

Less_1/ya
Less_1LessSum_1:output:0Less_1/y:output:0*
T0*
_output_shapes	
:ČL2
Less_1S
LogicalNot_1
LogicalNot
Less_1:z:0*
_output_shapes	
:ČL2
LogicalNot_1Y
	Squeeze_1SqueezeLogicalNot_1:y:0*
T0
*
_output_shapes	
:ČL2
	Squeeze_1k
boolean_mask_1/ShapeShapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2¨
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indicesŞ
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Prodo
boolean_mask_1/Shape_1Shapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2Ä
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1o
boolean_mask_1/Shape_2Shapetranspose_4:y:0*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2Ä
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2 
boolean_mask_1/strided_slice_2
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concat
boolean_mask_1/ReshapeReshapetranspose_4:y:0boolean_mask_1/concat:output:0*
T0*(
_output_shapes
:ČL˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Reshape
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2 
boolean_mask_1/Reshape_1/shape˘
boolean_mask_1/Reshape_1ReshapeSqueeze_1:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*
_output_shapes	
:ČL2
boolean_mask_1/Reshape_1
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
boolean_mask_1/Where
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
boolean_mask_1/GatherV2u
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_5/perm
transpose_5	Transpose boolean_mask_1/GatherV2:output:0transpose_5/perm:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
transpose_5l
IdentityIdentitytranspose_5:y:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ť
serving_default
;
input_10
serving_default_input_1:0˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:­3

biholomorphic_k4

layer1

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature"
_tf_keras_modelý{"class_name": "OuterProductNN_k4", "name": "outer_product_nn_k4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "OuterProductNN_k4"}}
ň
#	_self_saveable_object_factories

	variables
trainable_variables
regularization_losses
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"ž
_tf_keras_layer¤{"class_name": "Biholomorphic_k4", "name": "biholomorphic_k4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic_k4", "trainable": true, "dtype": "float32"}}
Ú
w
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "WidthOneDense", "name": "width_one_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
,
*serving_default"
signature_map
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
	variables
trainable_variables
layer_regularization_losses
layer_metrics
metrics

layers
regularization_losses
non_trainable_variables
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

	variables
trainable_variables
layer_regularization_losses
layer_metrics
metrics

layers
regularization_losses
non_trainable_variables
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
:	¤&2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables
trainable_variables
layer_regularization_losses
layer_metrics
 metrics

!layers
regularization_losses
"non_trainable_variables
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
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
ö2ó
2__inference_outer_product_nn_k4_layer_call_fn_1906ź
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
2
M__inference_outer_product_nn_k4_layer_call_and_return_conditional_losses_1882ź
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
ß2Ü
!__inference__wrapped_model_108844ś
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ď2Ě
/__inference_biholomorphic_k4_layer_call_fn_1911
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ę2ç
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_2328
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Î2Ë
.__inference_width_one_dense_layer_call_fn_1763
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
é2ć
I__inference_width_one_dense_layer_call_and_return_conditional_losses_2217
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
3B1
$__inference_signature_wrapper_108853input_1
!__inference__wrapped_model_108844j0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙Ż
J__inference_biholomorphic_k4_layer_call_and_return_conditional_losses_2328a/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
/__inference_biholomorphic_k4_layer_call_fn_1911T/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙­
M__inference_outer_product_nn_k4_layer_call_and_return_conditional_losses_1882\0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
2__inference_outer_product_nn_k4_layer_call_fn_1906O0˘-
&˘#
!
input_1˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
$__inference_signature_wrapper_108853u;˘8
˘ 
1Ş.
,
input_1!
input_1˙˙˙˙˙˙˙˙˙"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙ą
I__inference_width_one_dense_layer_call_and_return_conditional_losses_2217d8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
.__inference_width_one_dense_layer_call_fn_1763W8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙