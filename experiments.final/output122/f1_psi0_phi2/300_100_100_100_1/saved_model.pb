�
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
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18�
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	�*
dtype0
q

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*
shared_name
Variable_1
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�d*
dtype0
p

Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_name
Variable_2
i
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes

:dd*
dtype0
p

Variable_3VarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*
shared_name
Variable_3
i
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes

:dd*
dtype0
p

Variable_4VarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_name
Variable_4
i
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

:d*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
biholomorphic

layer1

layer2

layer3

layer4

layer5

signatures
#_self_saveable_object_factories
	trainable_variables

	variables
regularization_losses
	keras_api
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
~
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
~
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
~
w
#_self_saveable_object_factories
 trainable_variables
!	variables
"regularization_losses
#	keras_api
~
$w
#%_self_saveable_object_factories
&trainable_variables
'	variables
(regularization_losses
)	keras_api
~
*w
#+_self_saveable_object_factories
,trainable_variables
-	variables
.regularization_losses
/	keras_api
 
 
#
0
1
2
$3
*4
#
0
1
2
$3
*4
 
�
0non_trainable_variables

1layers
	trainable_variables

	variables
2layer_metrics
3metrics
4layer_regularization_losses
regularization_losses
 
 
 
 
�
5layer_regularization_losses
trainable_variables
	variables
regularization_losses
6layer_metrics
7metrics
8non_trainable_variables

9layers
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
�
:layer_regularization_losses
trainable_variables
	variables
regularization_losses
;layer_metrics
<metrics
=non_trainable_variables

>layers
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
�
?layer_regularization_losses
trainable_variables
	variables
regularization_losses
@layer_metrics
Ametrics
Bnon_trainable_variables

Clayers
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
�
Dlayer_regularization_losses
 trainable_variables
!	variables
"regularization_losses
Elayer_metrics
Fmetrics
Gnon_trainable_variables

Hlayers
CA
VARIABLE_VALUE
Variable_3#layer4/w/.ATTRIBUTES/VARIABLE_VALUE
 

$0

$0
 
�
Ilayer_regularization_losses
&trainable_variables
'	variables
(regularization_losses
Jlayer_metrics
Kmetrics
Lnon_trainable_variables

Mlayers
CA
VARIABLE_VALUE
Variable_4#layer5/w/.ATTRIBUTES/VARIABLE_VALUE
 

*0

*0
 
�
Nlayer_regularization_losses
,trainable_variables
-	variables
.regularization_losses
Olayer_metrics
Pmetrics
Qnon_trainable_variables

Rlayers
 
*
0
1
2
3
4
5
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
Variable_3
Variable_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_266109
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOpVariable_3/Read/ReadVariableOpVariable_4/Read/ReadVariableOpConst*
Tin
	2*
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
__inference__traced_save_266147
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1
Variable_2
Variable_3
Variable_4*
Tin

2*
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
"__inference__traced_restore_266172��
�
�
?__forward_dense_3_layer_call_and_return_conditional_losses_2413
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_2399_2414:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_2010

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
l
&__inference_dense_4_layer_call_fn_2002

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
GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_19962
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
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2054

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
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
:����������::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_dense_4_layer_call_and_return_conditional_losses_2107

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
�
�
D__inference_fourlayers_layer_call_and_return_conditional_losses_2090
input_1
dense_12030146
dense_1_12030166
dense_2_12030186
dense_3_12030206
dense_4_12030225
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapes�
�:������������������::2���������:���������:2���������:: :���������:���������:���������: : :���������:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19012
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_12030146*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_20682
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_12030166*
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
A__inference_dense_1_layer_call_and_return_conditional_losses_20542!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_12030186*
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
A__inference_dense_2_layer_call_and_return_conditional_losses_18112!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_12030206*
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
A__inference_dense_3_layer_call_and_return_conditional_losses_20402!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_12030225*
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
GPU2*0J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_19962!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Log�
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
?__forward_dense_1_layer_call_and_return_conditional_losses_3645
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*+
_input_shapes
:����������:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_3622_3646:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_266172
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3!
assignvariableop_4_variable_4

identity_6��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB#layer5/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

22
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
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5�

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�@
�
E__forward_biholomorphic_layer_call_and_return_conditional_losses_2562
inputs_0
identity
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
concat_axis
real
imag
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
conjG
ConjConjinputs_0*'
_output_shapes
:���������2
Conj�
einsum/EinsumEinsuminputs_0Conj:output:0*
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
transpose/permX
transpose_0	Transposeconcat:output:0transpose/perm:output:0*
T02
	transposeT
AbsAbstranspose_0:y:0*
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
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
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
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
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
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_0:y:0*
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
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
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

Identity"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:���������*s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_2495_2563:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__forward_dense_3_layer_call_and_return_conditional_losses_3585
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_3562_3586:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__forward_dense_4_layer_call_and_return_conditional_losses_2391
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_2381_2392:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
)__inference_fourlayers_layer_call_fn_2100
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_fourlayers_layer_call_and_return_conditional_losses_20902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
'__inference_restored_function_body_2355
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_fourlayers_layer_call_and_return_conditional_losses_20902
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
j
$__inference_dense_layer_call_fn_2074

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
:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_20682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
=__forward_dense_layer_call_and_return_conditional_losses_3675
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :������������������:*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_3652_3676:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_266109
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_2660922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
A__inference_dense_1_layer_call_and_return_conditional_losses_2115

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
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
:����������::P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
?__inference_dense_layer_call_and_return_conditional_losses_2068

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :������������������::X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
?__forward_dense_2_layer_call_and_return_conditional_losses_2437
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_2423_2438:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__forward_dense_4_layer_call_and_return_conditional_losses_3556
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_3540_3557:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�:
�
B__forward_fourlayers_layer_call_and_return_conditional_losses_2597
input_1
dense_12030146
dense_1_12030166
dense_2_12030186
dense_3_12030206
dense_4_12030225
identity#
dense_4_statefulpartitionedcall%
!dense_4_statefulpartitionedcall_0%
!dense_4_statefulpartitionedcall_1#
dense_3_statefulpartitionedcall%
!dense_3_statefulpartitionedcall_0%
!dense_3_statefulpartitionedcall_1#
dense_2_statefulpartitionedcall%
!dense_2_statefulpartitionedcall_0%
!dense_2_statefulpartitionedcall_1#
dense_1_statefulpartitionedcall%
!dense_1_statefulpartitionedcall_0%
!dense_1_statefulpartitionedcall_1!
dense_statefulpartitionedcall#
dense_statefulpartitionedcall_0#
dense_statefulpartitionedcall_1!
biholomorphic_partitionedcall#
biholomorphic_partitionedcall_0#
biholomorphic_partitionedcall_1	#
biholomorphic_partitionedcall_2#
biholomorphic_partitionedcall_3#
biholomorphic_partitionedcall_4#
biholomorphic_partitionedcall_5#
biholomorphic_partitionedcall_6#
biholomorphic_partitionedcall_7#
biholomorphic_partitionedcall_8	#
biholomorphic_partitionedcall_9	$
 biholomorphic_partitionedcall_10$
 biholomorphic_partitionedcall_11��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *�
_output_shapes�
�:������������������::2���������:���������:2���������:: :���������:���������:���������: : :���������:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_25622
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_12030146*
Tin
2*
Tout
2*
_collective_manager_ids
 *c
_output_shapesQ
O:����������:����������:	�:������������������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_24852
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_12030166*
Tin
2*
Tout
2*
_collective_manager_ids
 *Y
_output_shapesG
E:���������d:���������d:	�d:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_24612!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_12030186*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:���������d:���������d:dd:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_24372!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_12030206*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:���������d:���������d:dd:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_24132!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_12030225*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:���������:d:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_4_layer_call_and_return_conditional_losses_23912!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Log�
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"G
biholomorphic_partitionedcall&biholomorphic/PartitionedCall:output:1"I
biholomorphic_partitionedcall_0&biholomorphic/PartitionedCall:output:2"I
biholomorphic_partitionedcall_1&biholomorphic/PartitionedCall:output:3"K
 biholomorphic_partitionedcall_10'biholomorphic/PartitionedCall:output:12"K
 biholomorphic_partitionedcall_11'biholomorphic/PartitionedCall:output:13"I
biholomorphic_partitionedcall_2&biholomorphic/PartitionedCall:output:4"I
biholomorphic_partitionedcall_3&biholomorphic/PartitionedCall:output:5"I
biholomorphic_partitionedcall_4&biholomorphic/PartitionedCall:output:6"I
biholomorphic_partitionedcall_5&biholomorphic/PartitionedCall:output:7"I
biholomorphic_partitionedcall_6&biholomorphic/PartitionedCall:output:8"I
biholomorphic_partitionedcall_7&biholomorphic/PartitionedCall:output:9"J
biholomorphic_partitionedcall_8'biholomorphic/PartitionedCall:output:10"J
biholomorphic_partitionedcall_9'biholomorphic/PartitionedCall:output:11"K
dense_1_statefulpartitionedcall(dense_1/StatefulPartitionedCall:output:1"M
!dense_1_statefulpartitionedcall_0(dense_1/StatefulPartitionedCall:output:2"M
!dense_1_statefulpartitionedcall_1(dense_1/StatefulPartitionedCall:output:3"K
dense_2_statefulpartitionedcall(dense_2/StatefulPartitionedCall:output:1"M
!dense_2_statefulpartitionedcall_0(dense_2/StatefulPartitionedCall:output:2"M
!dense_2_statefulpartitionedcall_1(dense_2/StatefulPartitionedCall:output:3"K
dense_3_statefulpartitionedcall(dense_3/StatefulPartitionedCall:output:1"M
!dense_3_statefulpartitionedcall_0(dense_3/StatefulPartitionedCall:output:2"M
!dense_3_statefulpartitionedcall_1(dense_3/StatefulPartitionedCall:output:3"K
dense_4_statefulpartitionedcall(dense_4/StatefulPartitionedCall:output:0"M
!dense_4_statefulpartitionedcall_0(dense_4/StatefulPartitionedCall:output:1"M
!dense_4_statefulpartitionedcall_1(dense_4/StatefulPartitionedCall:output:2"G
dense_statefulpartitionedcall&dense/StatefulPartitionedCall:output:1"I
dense_statefulpartitionedcall_0&dense/StatefulPartitionedCall:output:2"I
dense_statefulpartitionedcall_1&dense/StatefulPartitionedCall:output:3"
identityIdentity:output:0*:
_input_shapes)
':���������:::::*p
backward_function_nameVT__inference___backward_fourlayers_layer_call_and_return_conditional_losses_2374_25982>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
?__forward_dense_2_layer_call_and_return_conditional_losses_3615
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:���������d:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_3592_3616:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
l
&__inference_dense_3_layer_call_fn_2046

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
A__inference_dense_3_layer_call_and_return_conditional_losses_20402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

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
&__inference_dense_1_layer_call_fn_2060

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
A__inference_dense_1_layer_call_and_return_conditional_losses_20542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
H
,__inference_biholomorphic_layer_call_fn_1906

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
GPU2*0J 8� *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19012
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
A__inference_dense_2_layer_call_and_return_conditional_losses_1811

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__forward_dense_1_layer_call_and_return_conditional_losses_2461
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
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
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*+
_input_shapes
:����������:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_2447_2462:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1901

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
�
�
A__inference_dense_3_layer_call_and_return_conditional_losses_2123

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
l
&__inference_dense_2_layer_call_fn_1817

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
A__inference_dense_2_layer_call_and_return_conditional_losses_18112
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0**
_input_shapes
:���������d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
!__inference__wrapped_model_266092
input_1
fourlayers_266080
fourlayers_266082
fourlayers_266084
fourlayers_266086
fourlayers_266088
identity��"fourlayers/StatefulPartitionedCall�
"fourlayers/StatefulPartitionedCallStatefulPartitionedCallinput_1fourlayers_266080fourlayers_266082fourlayers_266084fourlayers_266086fourlayers_266088*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_restored_function_body_23552$
"fourlayers/StatefulPartitionedCall�
IdentityIdentity+fourlayers/StatefulPartitionedCall:output:0#^fourlayers/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::2H
"fourlayers/StatefulPartitionedCall"fourlayers/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
=__forward_dense_layer_call_and_return_conditional_losses_2485
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputs��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:����������2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :������������������:*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_2471_2486:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�@
�
E__forward_biholomorphic_layer_call_and_return_conditional_losses_3769
inputs_0
identity
transpose_1_perm
boolean_mask_reshape
boolean_mask_squeeze	
	transpose
transpose_perm
concat_axis
real
imag
matrixbandpart
matrixbandpart_num_lower	
matrixbandpart_num_upper	

inputs
conjG
ConjConjinputs_0*'
_output_shapes
:���������2
Conj�
einsum/EinsumEinsuminputs_0Conj:output:0*
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
transpose/permX
transpose_0	Transposeconcat:output:0transpose/perm:output:0*
T02
	transposeT
AbsAbstranspose_0:y:0*
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
Squeezeg
boolean_mask/ShapeShapetranspose_0:y:0*
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
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
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
boolean_mask/strided_slice_1k
boolean_mask/Shape_2Shapetranspose_0:y:0*
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
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
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

Identity"5
boolean_mask_reshapeboolean_mask/Reshape:output:0"5
boolean_mask_squeezeboolean_mask/Squeeze:output:0"#
concat_axisconcat/axis:output:0"
conjConj:output:0"
identityIdentity:output:0"
imagImag:output:0"
inputsinputs_0"'
matrixbandpartMatrixBandPart:band:0"=
matrixbandpart_num_lower!MatrixBandPart/num_lower:output:0"=
matrixbandpart_num_upper!MatrixBandPart/num_upper:output:0"
realReal:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0")
transpose_permtranspose/perm:output:0*&
_input_shapes
:���������*s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_3682_3770:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference__traced_save_266147
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop
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
value3B1 B+_temp_6097bf96c61e4691a8113ef74288763f/part2	
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
:*
dtype0*�
value�B�B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB#layer5/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
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

identity_1Identity_1:output:0*K
_input_shapes:
8: :	�:	�d:dd:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:%!

_output_shapes
:	�d:$ 

_output_shapes

:dd:$ 

_output_shapes

:dd:$ 

_output_shapes

:d:

_output_shapes
: 
�
�
A__inference_dense_3_layer_call_and_return_conditional_losses_2040

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1803

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
�:
�
B__forward_fourlayers_layer_call_and_return_conditional_losses_3791
input_1
dense_12030146
dense_1_12030166
dense_2_12030186
dense_3_12030206
dense_4_12030225
identity#
dense_4_statefulpartitionedcall%
!dense_4_statefulpartitionedcall_0%
!dense_4_statefulpartitionedcall_1#
dense_3_statefulpartitionedcall%
!dense_3_statefulpartitionedcall_0%
!dense_3_statefulpartitionedcall_1#
dense_2_statefulpartitionedcall%
!dense_2_statefulpartitionedcall_0%
!dense_2_statefulpartitionedcall_1#
dense_1_statefulpartitionedcall%
!dense_1_statefulpartitionedcall_0%
!dense_1_statefulpartitionedcall_1!
dense_statefulpartitionedcall#
dense_statefulpartitionedcall_0#
dense_statefulpartitionedcall_1!
biholomorphic_partitionedcall#
biholomorphic_partitionedcall_0#
biholomorphic_partitionedcall_1	#
biholomorphic_partitionedcall_2#
biholomorphic_partitionedcall_3#
biholomorphic_partitionedcall_4#
biholomorphic_partitionedcall_5#
biholomorphic_partitionedcall_6#
biholomorphic_partitionedcall_7#
biholomorphic_partitionedcall_8	#
biholomorphic_partitionedcall_9	$
 biholomorphic_partitionedcall_10$
 biholomorphic_partitionedcall_11��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *�
_output_shapes�
�:������������������::2���������:���������:2���������:: :���������:���������:���������: : :���������:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_37692
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_12030146*
Tin
2*
Tout
2*
_collective_manager_ids
 *c
_output_shapesQ
O:����������:����������:	�:������������������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_36752
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_12030166*
Tin
2*
Tout
2*
_collective_manager_ids
 *Y
_output_shapesG
E:���������d:���������d:	�d:����������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_36452!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_12030186*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:���������d:���������d:dd:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_36152!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_12030206*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:���������d:���������d:dd:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_35852!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_12030225*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:���������:d:���������d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__forward_dense_4_layer_call_and_return_conditional_losses_35562!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
Log�
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"G
biholomorphic_partitionedcall&biholomorphic/PartitionedCall:output:1"I
biholomorphic_partitionedcall_0&biholomorphic/PartitionedCall:output:2"I
biholomorphic_partitionedcall_1&biholomorphic/PartitionedCall:output:3"K
 biholomorphic_partitionedcall_10'biholomorphic/PartitionedCall:output:12"K
 biholomorphic_partitionedcall_11'biholomorphic/PartitionedCall:output:13"I
biholomorphic_partitionedcall_2&biholomorphic/PartitionedCall:output:4"I
biholomorphic_partitionedcall_3&biholomorphic/PartitionedCall:output:5"I
biholomorphic_partitionedcall_4&biholomorphic/PartitionedCall:output:6"I
biholomorphic_partitionedcall_5&biholomorphic/PartitionedCall:output:7"I
biholomorphic_partitionedcall_6&biholomorphic/PartitionedCall:output:8"I
biholomorphic_partitionedcall_7&biholomorphic/PartitionedCall:output:9"J
biholomorphic_partitionedcall_8'biholomorphic/PartitionedCall:output:10"J
biholomorphic_partitionedcall_9'biholomorphic/PartitionedCall:output:11"K
dense_1_statefulpartitionedcall(dense_1/StatefulPartitionedCall:output:1"M
!dense_1_statefulpartitionedcall_0(dense_1/StatefulPartitionedCall:output:2"M
!dense_1_statefulpartitionedcall_1(dense_1/StatefulPartitionedCall:output:3"K
dense_2_statefulpartitionedcall(dense_2/StatefulPartitionedCall:output:1"M
!dense_2_statefulpartitionedcall_0(dense_2/StatefulPartitionedCall:output:2"M
!dense_2_statefulpartitionedcall_1(dense_2/StatefulPartitionedCall:output:3"K
dense_3_statefulpartitionedcall(dense_3/StatefulPartitionedCall:output:1"M
!dense_3_statefulpartitionedcall_0(dense_3/StatefulPartitionedCall:output:2"M
!dense_3_statefulpartitionedcall_1(dense_3/StatefulPartitionedCall:output:3"K
dense_4_statefulpartitionedcall(dense_4/StatefulPartitionedCall:output:0"M
!dense_4_statefulpartitionedcall_0(dense_4/StatefulPartitionedCall:output:1"M
!dense_4_statefulpartitionedcall_1(dense_4/StatefulPartitionedCall:output:2"G
dense_statefulpartitionedcall&dense/StatefulPartitionedCall:output:1"I
dense_statefulpartitionedcall_0&dense/StatefulPartitionedCall:output:2"I
dense_statefulpartitionedcall_1&dense/StatefulPartitionedCall:output:3"
identityIdentity:output:0*:
_input_shapes)
':���������:::::*p
backward_function_nameVT__inference___backward_fourlayers_layer_call_and_return_conditional_losses_3488_37922>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
A__inference_dense_2_layer_call_and_return_conditional_losses_1848

inputs"
matmul_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
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
identityIdentity:output:0**
_input_shapes
:���������d::O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
A__inference_dense_4_layer_call_and_return_conditional_losses_1996

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
StatefulPartitionedCall:0���������tensorflow/serving/predict:�l
�
biholomorphic

layer1

layer2

layer3

layer4

layer5

signatures
#_self_saveable_object_factories
	trainable_variables

	variables
regularization_losses
	keras_api
S__call__
T_default_save_signature
*U&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "fourlayers", "name": "fourlayers", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "fourlayers"}}
�
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
�
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
#_self_saveable_object_factories
 trainable_variables
!	variables
"regularization_losses
#	keras_api
\__call__
*]&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
$w
#%_self_saveable_object_factories
&trainable_variables
'	variables
(regularization_losses
)	keras_api
^__call__
*_&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
*w
#+_self_saveable_object_factories
,trainable_variables
-	variables
.regularization_losses
/	keras_api
`__call__
*a&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
,
bserving_default"
signature_map
 "
trackable_dict_wrapper
C
0
1
2
$3
*4"
trackable_list_wrapper
C
0
1
2
$3
*4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0non_trainable_variables

1layers
	trainable_variables

	variables
2layer_metrics
3metrics
4layer_regularization_losses
regularization_losses
S__call__
T_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
5layer_regularization_losses
trainable_variables
	variables
regularization_losses
6layer_metrics
7metrics
8non_trainable_variables

9layers
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	�2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
:layer_regularization_losses
trainable_variables
	variables
regularization_losses
;layer_metrics
<metrics
=non_trainable_variables

>layers
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
:	�d2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?layer_regularization_losses
trainable_variables
	variables
regularization_losses
@layer_metrics
Ametrics
Bnon_trainable_variables

Clayers
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dlayer_regularization_losses
 trainable_variables
!	variables
"regularization_losses
Elayer_metrics
Fmetrics
Gnon_trainable_variables

Hlayers
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
'
$0"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ilayer_regularization_losses
&trainable_variables
'	variables
(regularization_losses
Jlayer_metrics
Kmetrics
Lnon_trainable_variables

Mlayers
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
 "
trackable_dict_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nlayer_regularization_losses
,trainable_variables
-	variables
.regularization_losses
Olayer_metrics
Pmetrics
Qnon_trainable_variables

Rlayers
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
)__inference_fourlayers_layer_call_fn_2100�
���
FullArgSpec
args�

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
annotations� *&�#
!�
input_1���������
�2�
!__inference__wrapped_model_266092�
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
D__inference_fourlayers_layer_call_and_return_conditional_losses_2090�
���
FullArgSpec
args�

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
annotations� *&�#
!�
input_1���������
�2�
,__inference_biholomorphic_layer_call_fn_1906�
���
FullArgSpec
args�

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
annotations� *
 
�2�
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1803�
���
FullArgSpec
args�

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
annotations� *
 
�2�
$__inference_dense_layer_call_fn_2074�
���
FullArgSpec
args�

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
annotations� *
 
�2�
?__inference_dense_layer_call_and_return_conditional_losses_2010�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_1_layer_call_fn_2060�
���
FullArgSpec
args�

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
annotations� *
 
�2�
A__inference_dense_1_layer_call_and_return_conditional_losses_2115�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_2_layer_call_fn_1817�
���
FullArgSpec
args�

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
annotations� *
 
�2�
A__inference_dense_2_layer_call_and_return_conditional_losses_1848�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_3_layer_call_fn_2046�
���
FullArgSpec
args�

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
annotations� *
 
�2�
A__inference_dense_3_layer_call_and_return_conditional_losses_2123�
���
FullArgSpec
args�

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
annotations� *
 
�2�
&__inference_dense_4_layer_call_fn_2002�
���
FullArgSpec
args�

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
annotations� *
 
�2�
A__inference_dense_4_layer_call_and_return_conditional_losses_2107�
���
FullArgSpec
args�

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
annotations� *
 
3B1
$__inference_signature_wrapper_266109input_1�
!__inference__wrapped_model_266092n$*0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1803a/�,
%�"
 �
inputs���������
� ".�+
$�!
0������������������
� �
,__inference_biholomorphic_layer_call_fn_1906T/�,
%�"
 �
inputs���������
� "!��������������������
A__inference_dense_1_layer_call_and_return_conditional_losses_2115\0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� y
&__inference_dense_1_layer_call_fn_2060O0�-
&�#
!�
inputs����������
� "����������d�
A__inference_dense_2_layer_call_and_return_conditional_losses_1848[/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� x
&__inference_dense_2_layer_call_fn_1817N/�,
%�"
 �
inputs���������d
� "����������d�
A__inference_dense_3_layer_call_and_return_conditional_losses_2123[$/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� x
&__inference_dense_3_layer_call_fn_2046N$/�,
%�"
 �
inputs���������d
� "����������d�
A__inference_dense_4_layer_call_and_return_conditional_losses_2107[*/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� x
&__inference_dense_4_layer_call_fn_2002N*/�,
%�"
 �
inputs���������d
� "�����������
?__inference_dense_layer_call_and_return_conditional_losses_2010e8�5
.�+
)�&
inputs������������������
� "&�#
�
0����������
� �
$__inference_dense_layer_call_fn_2074X8�5
.�+
)�&
inputs������������������
� "������������
D__inference_fourlayers_layer_call_and_return_conditional_losses_2090`$*0�-
&�#
!�
input_1���������
� "%�"
�
0���������
� �
)__inference_fourlayers_layer_call_fn_2100S$*0�-
&�#
!�
input_1���������
� "�����������
$__inference_signature_wrapper_266109y$*;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������