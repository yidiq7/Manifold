ѥ
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
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
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
		variables

regularization_losses
trainable_variables
	keras_api
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
~
$w
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
~
*w
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api
 
 
#
0
1
2
$3
*4
 
#
0
1
2
$3
*4
�

0layers
		variables
1non_trainable_variables

regularization_losses
trainable_variables
2layer_regularization_losses
3metrics
4layer_metrics
 
 
 
 
�

5layers
	variables
6non_trainable_variables
regularization_losses
trainable_variables
7layer_regularization_losses
8metrics
9layer_metrics
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
�

:layers
	variables
;non_trainable_variables
regularization_losses
trainable_variables
<layer_regularization_losses
=metrics
>layer_metrics
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
�

?layers
	variables
@non_trainable_variables
regularization_losses
trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE
 

0
 

0
�

Dlayers
 	variables
Enon_trainable_variables
!regularization_losses
"trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
CA
VARIABLE_VALUE
Variable_3#layer4/w/.ATTRIBUTES/VARIABLE_VALUE
 

$0
 

$0
�

Ilayers
&	variables
Jnon_trainable_variables
'regularization_losses
(trainable_variables
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
CA
VARIABLE_VALUE
Variable_4#layer5/w/.ATTRIBUTES/VARIABLE_VALUE
 

*0
 

*0
�

Nlayers
,	variables
Onon_trainable_variables
-regularization_losses
.trainable_variables
Player_regularization_losses
Qmetrics
Rlayer_metrics
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
GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_15929433
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
GPU2*0J 8� **
f%R#
!__inference__traced_save_15929471
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
GPU2*0J 8� *-
f(R&
$__inference__traced_restore_15929496Ɇ
�
�
/__inference_kahler_potential_layer_call_fn_2334
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
GPU2*0J 8� *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_22942
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
A__inference_dense_1_layer_call_and_return_conditional_losses_2131

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
�
�
?__forward_dense_4_layer_call_and_return_conditional_losses_3688
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
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_3672_3689:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__forward_dense_1_layer_call_and_return_conditional_losses_2593
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
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_2579_2594:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294
input_1
dense_73283571
dense_1_73283591
dense_2_73283611
dense_3_73283631
dense_4_73283650
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
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19142
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
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
?__inference_dense_layer_call_and_return_conditional_losses_22782
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
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
A__inference_dense_1_layer_call_and_return_conditional_losses_21312!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
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
A__inference_dense_2_layer_call_and_return_conditional_losses_19572!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
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
A__inference_dense_3_layer_call_and_return_conditional_losses_17792!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
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
A__inference_dense_4_layer_call_and_return_conditional_losses_17572!
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
�
�
$__inference__traced_restore_15929496
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
�
�
?__forward_dense_3_layer_call_and_return_conditional_losses_2545
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
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_2531_2546:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
!__inference__traced_save_15929471
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
value3B1 B+_temp_63a4f8159cb6431fbaa4c28a1babb756/part2	
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
A__inference_dense_4_layer_call_and_return_conditional_losses_1757

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
�
�
&__inference_signature_wrapper_15929433
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
GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_159294162
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
�@
�
E__forward_biholomorphic_layer_call_and_return_conditional_losses_3901
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
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_3814_3902:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__forward_dense_1_layer_call_and_return_conditional_losses_3777
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
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_3754_3778:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�:
�
H__forward_kahler_potential_layer_call_and_return_conditional_losses_2729
input_1
dense_73283571
dense_1_73283591
dense_2_73283611
dense_3_73283631
dense_4_73283650
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
E__forward_biholomorphic_layer_call_and_return_conditional_losses_26942
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
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
=__forward_dense_layer_call_and_return_conditional_losses_26172
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
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
?__forward_dense_1_layer_call_and_return_conditional_losses_25932!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
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
?__forward_dense_2_layer_call_and_return_conditional_losses_25692!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
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
?__forward_dense_3_layer_call_and_return_conditional_losses_25452!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
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
?__forward_dense_4_layer_call_and_return_conditional_losses_25232!
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
':���������:::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_2506_27302>
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
?__forward_dense_2_layer_call_and_return_conditional_losses_3747
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
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_3724_3748:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1914

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
�
j
$__inference_dense_layer_call_fn_2340

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
?__inference_dense_layer_call_and_return_conditional_losses_22782
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
�
l
&__inference_dense_4_layer_call_fn_1763

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
A__inference_dense_4_layer_call_and_return_conditional_losses_17572
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
A__inference_dense_3_layer_call_and_return_conditional_losses_1779

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
�
�
=__forward_dense_layer_call_and_return_conditional_losses_2617
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
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_2603_2618:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
A__inference_dense_2_layer_call_and_return_conditional_losses_2161

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
A__inference_dense_4_layer_call_and_return_conditional_losses_1970

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
A__inference_dense_2_layer_call_and_return_conditional_losses_1957

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
&__inference_dense_2_layer_call_fn_1963

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
A__inference_dense_2_layer_call_and_return_conditional_losses_19572
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
�
�
'__inference_restored_function_body_2487
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
GPU2*0J 8� *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_22942
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
A__inference_dense_1_layer_call_and_return_conditional_losses_2070

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
?__inference_dense_layer_call_and_return_conditional_losses_2270

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
�	
�
#__inference__wrapped_model_15929416
input_1
kahler_potential_15929404
kahler_potential_15929406
kahler_potential_15929408
kahler_potential_15929410
kahler_potential_15929412
identity��(kahler_potential/StatefulPartitionedCall�
(kahler_potential/StatefulPartitionedCallStatefulPartitionedCallinput_1kahler_potential_15929404kahler_potential_15929406kahler_potential_15929408kahler_potential_15929410kahler_potential_15929412*
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
'__inference_restored_function_body_24872*
(kahler_potential/StatefulPartitionedCall�
IdentityIdentity1kahler_potential/StatefulPartitionedCall:output:0)^kahler_potential/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������:::::2T
(kahler_potential/StatefulPartitionedCall(kahler_potential/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
?__inference_dense_layer_call_and_return_conditional_losses_2278

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
&__inference_dense_1_layer_call_fn_2137

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
A__inference_dense_1_layer_call_and_return_conditional_losses_21312
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
�
l
&__inference_dense_3_layer_call_fn_1785

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
A__inference_dense_3_layer_call_and_return_conditional_losses_17792
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
�
�
?__forward_dense_2_layer_call_and_return_conditional_losses_2569
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
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_2555_2570:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
?__forward_dense_3_layer_call_and_return_conditional_losses_3717
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
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_3694_3718:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_2123

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
�@
�
E__forward_biholomorphic_layer_call_and_return_conditional_losses_2694
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
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_2627_2695:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
?__forward_dense_4_layer_call_and_return_conditional_losses_2523
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
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_2513_2524:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�:
�
H__forward_kahler_potential_layer_call_and_return_conditional_losses_3923
input_1
dense_73283571
dense_1_73283591
dense_2_73283611
dense_3_73283631
dense_4_73283650
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
E__forward_biholomorphic_layer_call_and_return_conditional_losses_39012
biholomorphic/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
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
=__forward_dense_layer_call_and_return_conditional_losses_38072
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
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
?__forward_dense_1_layer_call_and_return_conditional_losses_37772!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
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
?__forward_dense_2_layer_call_and_return_conditional_losses_37472!
dense_2/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
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
?__forward_dense_3_layer_call_and_return_conditional_losses_37172!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
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
?__forward_dense_4_layer_call_and_return_conditional_losses_36882!
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
':���������:::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_3620_39242>
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
�
�
=__forward_dense_layer_call_and_return_conditional_losses_3807
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
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_3784_3808:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
�
A__inference_dense_3_layer_call_and_return_conditional_losses_2153

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
H
,__inference_biholomorphic_layer_call_fn_1919

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
G__inference_biholomorphic_layer_call_and_return_conditional_losses_19142
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
		variables

regularization_losses
trainable_variables
	keras_api
S_default_save_signature
*T&call_and_return_all_conditional_losses
U__call__"�
_tf_keras_model�{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
�
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"�
_tf_keras_layer�{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
�
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
#_self_saveable_object_factories
	variables
regularization_losses
trainable_variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
w
#_self_saveable_object_factories
 	variables
!regularization_losses
"trainable_variables
#	keras_api
*\&call_and_return_all_conditional_losses
]__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
$w
#%_self_saveable_object_factories
&	variables
'regularization_losses
(trainable_variables
)	keras_api
*^&call_and_return_all_conditional_losses
___call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
*w
#+_self_saveable_object_factories
,	variables
-regularization_losses
.trainable_variables
/	keras_api
*`&call_and_return_all_conditional_losses
a__call__"�
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
 "
trackable_list_wrapper
C
0
1
2
$3
*4"
trackable_list_wrapper
�

0layers
		variables
1non_trainable_variables

regularization_losses
trainable_variables
2layer_regularization_losses
3metrics
4layer_metrics
U__call__
S_default_save_signature
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
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

5layers
	variables
6non_trainable_variables
regularization_losses
trainable_variables
7layer_regularization_losses
8metrics
9layer_metrics
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	�2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�

:layers
	variables
;non_trainable_variables
regularization_losses
trainable_variables
<layer_regularization_losses
=metrics
>layer_metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
:	�d2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�

?layers
	variables
@non_trainable_variables
regularization_losses
trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
�

Dlayers
 	variables
Enon_trainable_variables
!regularization_losses
"trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
�

Ilayers
&	variables
Jnon_trainable_variables
'regularization_losses
(trainable_variables
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
 "
trackable_dict_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
�

Nlayers
,	variables
Onon_trainable_variables
-regularization_losses
.trainable_variables
Player_regularization_losses
Qmetrics
Rlayer_metrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
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
�2�
#__inference__wrapped_model_15929416�
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
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294�
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
/__inference_kahler_potential_layer_call_fn_2334�
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
G__inference_biholomorphic_layer_call_and_return_conditional_losses_2123�
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
,__inference_biholomorphic_layer_call_fn_1919�
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
?__inference_dense_layer_call_and_return_conditional_losses_2270�
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
$__inference_dense_layer_call_fn_2340�
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
A__inference_dense_1_layer_call_and_return_conditional_losses_2070�
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
&__inference_dense_1_layer_call_fn_2137�
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
A__inference_dense_2_layer_call_and_return_conditional_losses_2161�
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
&__inference_dense_2_layer_call_fn_1963�
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
A__inference_dense_3_layer_call_and_return_conditional_losses_2153�
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
&__inference_dense_3_layer_call_fn_1785�
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
A__inference_dense_4_layer_call_and_return_conditional_losses_1970�
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
&__inference_dense_4_layer_call_fn_1763�
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
5B3
&__inference_signature_wrapper_15929433input_1�
#__inference__wrapped_model_15929416n$*0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
G__inference_biholomorphic_layer_call_and_return_conditional_losses_2123a/�,
%�"
 �
inputs���������
� ".�+
$�!
0������������������
� �
,__inference_biholomorphic_layer_call_fn_1919T/�,
%�"
 �
inputs���������
� "!��������������������
A__inference_dense_1_layer_call_and_return_conditional_losses_2070\0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� y
&__inference_dense_1_layer_call_fn_2137O0�-
&�#
!�
inputs����������
� "����������d�
A__inference_dense_2_layer_call_and_return_conditional_losses_2161[/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� x
&__inference_dense_2_layer_call_fn_1963N/�,
%�"
 �
inputs���������d
� "����������d�
A__inference_dense_3_layer_call_and_return_conditional_losses_2153[$/�,
%�"
 �
inputs���������d
� "%�"
�
0���������d
� x
&__inference_dense_3_layer_call_fn_1785N$/�,
%�"
 �
inputs���������d
� "����������d�
A__inference_dense_4_layer_call_and_return_conditional_losses_1970[*/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� x
&__inference_dense_4_layer_call_fn_1763N*/�,
%�"
 �
inputs���������d
� "�����������
?__inference_dense_layer_call_and_return_conditional_losses_2270e8�5
.�+
)�&
inputs������������������
� "&�#
�
0����������
� �
$__inference_dense_layer_call_fn_2340X8�5
.�+
)�&
inputs������������������
� "������������
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294`$*0�-
&�#
!�
input_1���������
� "%�"
�
0���������
� �
/__inference_kahler_potential_layer_call_fn_2334S$*0�-
&�#
!�
input_1���������
� "�����������
&__inference_signature_wrapper_15929433y$*;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1���������