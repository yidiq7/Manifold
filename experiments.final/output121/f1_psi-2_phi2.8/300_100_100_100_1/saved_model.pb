ее
═г
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
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18Д▒
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	м*
dtype0
q

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	мd*
shared_name
Variable_1
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	мd*
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
И
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*├
value╣B╢ Bп
╓
biholomorphic

layer1

layer2

layer3

layer4

layer5

signatures
#_self_saveable_object_factories
	regularization_losses

	variables
trainable_variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
~
w
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api
~
$w
#%_self_saveable_object_factories
&regularization_losses
'	variables
(trainable_variables
)	keras_api
~
*w
#+_self_saveable_object_factories
,regularization_losses
-	variables
.trainable_variables
/	keras_api
 
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
н
	regularization_losses
0metrics

	variables
1non_trainable_variables

2layers
3layer_regularization_losses
trainable_variables
4layer_metrics
 
 
 
 
н
regularization_losses
5metrics
	variables
6non_trainable_variables

7layers
8layer_regularization_losses
trainable_variables
9layer_metrics
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
н
regularization_losses
:metrics
	variables
;non_trainable_variables

<layers
=layer_regularization_losses
trainable_variables
>layer_metrics
CA
VARIABLE_VALUE
Variable_1#layer2/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
н
regularization_losses
?metrics
	variables
@non_trainable_variables

Alayers
Blayer_regularization_losses
trainable_variables
Clayer_metrics
CA
VARIABLE_VALUE
Variable_2#layer3/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

0

0
н
 regularization_losses
Dmetrics
!	variables
Enon_trainable_variables

Flayers
Glayer_regularization_losses
"trainable_variables
Hlayer_metrics
CA
VARIABLE_VALUE
Variable_3#layer4/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

$0

$0
н
&regularization_losses
Imetrics
'	variables
Jnon_trainable_variables

Klayers
Llayer_regularization_losses
(trainable_variables
Mlayer_metrics
CA
VARIABLE_VALUE
Variable_4#layer5/w/.ATTRIBUTES/VARIABLE_VALUE
 
 

*0

*0
н
,regularization_losses
Nmetrics
-	variables
Onon_trainable_variables

Players
Qlayer_regularization_losses
.trainable_variables
Rlayer_metrics
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
∙
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
:         *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_524818
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_524856
╪
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_524881лЖ
ж:
Ц
H__forward_kahler_potential_layer_call_and_return_conditional_losses_4008
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
 biholomorphic_partitionedcall_11Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallж
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *▄
_output_shapes╔
╞:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_39862
biholomorphic/PartitionedCall╒
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
Tin
2*
Tout
2*
_collective_manager_ids
 *c
_output_shapesQ
O:         м:         м:	м:                  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_38922
dense/StatefulPartitionedCall╙
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
Tin
2*
Tout
2*
_collective_manager_ids
 *Y
_output_shapesG
E:         d:         d:	мd:         м*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_38622!
dense_1/StatefulPartitionedCall╙
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         d:         d:dd:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_38322!
dense_2/StatefulPartitionedCall╙
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         d:         d:dd:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_38022!
dense_3/StatefulPartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:         :d:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_4_layer_call_and_return_conditional_losses_37732!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
LogГ
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
':         :::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_3705_40092>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╣
l
&__inference_dense_3_layer_call_fn_1970

inputs
unknown
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ш
Д
A__inference_dense_3_layer_call_and_return_conditional_losses_1840

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
═
┴
__inference__traced_save_524856
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop)
%savev2_variable_2_read_readvariableop)
%savev2_variable_3_read_readvariableop)
%savev2_variable_4_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_49fb0f039e6045eebe993c2758c75992/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┌
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ь
valueтB▀B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB#layer5/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slicesА
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop%savev2_variable_3_read_readvariableop%savev2_variable_4_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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
8: :	м:	мd:dd:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	м:%!

_output_shapes
:	мd:$ 

_output_shapes

:dd:$ 

_output_shapes

:dd:$ 

_output_shapes

:d:

_output_shapes
: 
№
В
?__inference_dense_layer_call_and_return_conditional_losses_1933

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:         м2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  ::X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
ў
╖
?__forward_dense_1_layer_call_and_return_conditional_losses_2678
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	мd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*+
_input_shapes
:         м:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_2664_2679:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
О
Д
A__inference_dense_4_layer_call_and_return_conditional_losses_1948

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
┐
■
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294
input_1
dense_73283571
dense_1_73283591
dense_2_73283611
dense_3_73283631
dense_4_73283650
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallЫ
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *▄
_output_shapes╔
╞:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_21062
biholomorphic/PartitionedCallЩ
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_21502
dense/StatefulPartitionedCallа
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_17662!
dense_1/StatefulPartitionedCallв
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_21642!
dense_2/StatefulPartitionedCallв
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_3_layer_call_and_return_conditional_losses_19642!
dense_3/StatefulPartitionedCallв
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_22782!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
LogГ
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
и
к
'__inference_restored_function_body_2572
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*'
_output_shapes
:         *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_22942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         :::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
О
Д
A__inference_dense_4_layer_call_and_return_conditional_losses_2278

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╣
l
&__inference_dense_2_layer_call_fn_2170

inputs
unknown
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_21642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ї
╖
?__forward_dense_2_layer_call_and_return_conditional_losses_3832
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_3809_3833:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Є
л
?__forward_dense_4_layer_call_and_return_conditional_losses_2608
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_2598_2609:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ї
╖
?__forward_dense_2_layer_call_and_return_conditional_losses_2654
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_2_layer_call_and_return_conditional_losses_2640_2655:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╦9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_2106

inputs
identityE
ConjConjinputs*'
_output_shapes
:         2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upper╞
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
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
:2         2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2         2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesв
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2╕
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2╕
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axis№
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis√
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Є
л
?__forward_dense_4_layer_call_and_return_conditional_losses_3773
inputs_0"
matmul_readvariableop_resource
identity
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_4_layer_call_and_return_conditional_losses_3757_3774:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ш
Д
A__inference_dense_2_layer_call_and_return_conditional_losses_1925

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ж
╡
=__forward_dense_layer_call_and_return_conditional_losses_2702
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:         м2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :                  :*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_2688_2703:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
л@
█
E__forward_biholomorphic_layer_call_and_return_conditional_losses_3986
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
:         2
ConjЪ
einsum/EinsumEinsuminputs_0Conj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upper╞
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
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
:2         2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesв
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2╕
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2╕
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axis№
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatШ
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis√
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

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
:         *s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_3899_3987:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш
Д
A__inference_dense_2_layer_call_and_return_conditional_losses_2164

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╧
▓
/__inference_kahler_potential_layer_call_fn_2304
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_kahler_potential_layer_call_and_return_conditional_losses_22942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         :::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
н
H
,__inference_biholomorphic_layer_call_fn_2111

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_biholomorphic_layer_call_and_return_conditional_losses_21062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш
Д
A__inference_dense_3_layer_call_and_return_conditional_losses_1964

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0**
_input_shapes
:         d::O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
╔
j
$__inference_dense_layer_call_fn_2156

inputs
unknown
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         м*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_21502
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  :22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
╣
l
&__inference_dense_4_layer_call_fn_2340

inputs
unknown
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_22782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0**
_input_shapes
:         d:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ж
╡
=__forward_dense_layer_call_and_return_conditional_losses_3892
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOpv
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:         м2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*3
_input_shapes"
 :                  :*k
backward_function_nameQO__inference___backward_dense_layer_call_and_return_conditional_losses_3869_3893:X T
0
_output_shapes
:                  
 
_user_specified_nameinputs
з
┌
"__inference__traced_restore_524881
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1!
assignvariableop_2_variable_2!
assignvariableop_3_variable_3!
assignvariableop_4_variable_4

identity_6ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ь
valueтB▀B#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB#layer2/w/.ATTRIBUTES/VARIABLE_VALUEB#layer3/w/.ATTRIBUTES/VARIABLE_VALUEB#layer4/w/.ATTRIBUTES/VARIABLE_VALUEB#layer5/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices╔
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

IdentityШ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1в
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2в
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3в
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4в
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_4Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╧

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5┴

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
╗
l
&__inference_dense_1_layer_call_fn_1772

inputs
unknown
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_17662
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*+
_input_shapes
:         м:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
Ї
╖
?__forward_dense_3_layer_call_and_return_conditional_losses_3802
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_3779_3803:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
Ы
з
$__inference_signature_wrapper_524818
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityИвStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_5248012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         :::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
л@
█
E__forward_biholomorphic_layer_call_and_return_conditional_losses_2779
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
:         2
ConjЪ
einsum/EinsumEinsuminputs_0Conj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upper╞
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
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
:2         2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesв
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodk
boolean_mask/Shape_1Shapetranspose_0:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2╕
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2╕
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axis№
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatШ
boolean_mask/ReshapeReshapetranspose_0:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis√
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

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
:         *s
backward_function_nameYW__inference___backward_biholomorphic_layer_call_and_return_conditional_losses_2712_2780:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╦9
c
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1893

inputs
identityE
ConjConjinputs*'
_output_shapes
:         2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:         *
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
         2
MatrixBandPart/num_upper╞
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:         2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:         2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:         2
RealO
ImagImagReshape:output:0*'
_output_shapes
:         2
Imag\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЛ
concatConcatV2Real:output:0Imag:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         22
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
:2         2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2         2
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
 *oГ:2
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
boolean_mask/ShapeО
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stackТ
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1Т
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2Ь
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_sliceФ
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indicesв
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prodi
boolean_mask/Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2
boolean_mask/Shape_1Т
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stackЦ
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1Ц
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2╕
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
boolean_mask/Shape_2Т
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stackЦ
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1Ц
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2╕
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
boolean_mask/strided_slice_2О
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
boolean_mask/concat/axis№
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2         2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:         2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis√
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  2
boolean_mask/GatherV2u
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/permЭ
transpose_1	Transposeboolean_mask/GatherV2:output:0transpose_1/perm:output:0*
T0*0
_output_shapes
:                  2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ы
Д
A__inference_dense_1_layer_call_and_return_conditional_losses_1917

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	мd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*+
_input_shapes
:         м::P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
ж:
Ц
H__forward_kahler_potential_layer_call_and_return_conditional_losses_2814
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
 biholomorphic_partitionedcall_11Ивdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallж
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2			*
_collective_manager_ids
 *▄
_output_shapes╔
╞:                  ::2         :         :2         :: :         :         :         : : :         :         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__forward_biholomorphic_layer_call_and_return_conditional_losses_27792
biholomorphic/PartitionedCall╒
dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0dense_73283571*
Tin
2*
Tout
2*
_collective_manager_ids
 *c
_output_shapesQ
O:         м:         м:	м:                  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *F
fAR?
=__forward_dense_layer_call_and_return_conditional_losses_27022
dense/StatefulPartitionedCall╙
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_73283591*
Tin
2*
Tout
2*
_collective_manager_ids
 *Y
_output_shapesG
E:         d:         d:	мd:         м*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_1_layer_call_and_return_conditional_losses_26782!
dense_1/StatefulPartitionedCall╙
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_73283611*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         d:         d:dd:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_2_layer_call_and_return_conditional_losses_26542!
dense_2/StatefulPartitionedCall╙
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_73283631*
Tin
2*
Tout
2*
_collective_manager_ids
 *W
_output_shapesE
C:         d:         d:dd:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_3_layer_call_and_return_conditional_losses_26302!
dense_3/StatefulPartitionedCall┐
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_73283650*
Tin
2*
Tout
2*
_collective_manager_ids
 *D
_output_shapes2
0:         :d:         d*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__forward_dense_4_layer_call_and_return_conditional_losses_26082!
dense_4/StatefulPartitionedCallm
LogLog(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:         2
LogГ
IdentityIdentityLog:y:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

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
':         :::::*v
backward_function_name\Z__inference___backward_kahler_potential_layer_call_and_return_conditional_losses_2591_28152>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
е	
¤
!__inference__wrapped_model_524801
input_1
kahler_potential_524789
kahler_potential_524791
kahler_potential_524793
kahler_potential_524795
kahler_potential_524797
identityИв(kahler_potential/StatefulPartitionedCallь
(kahler_potential/StatefulPartitionedCallStatefulPartitionedCallinput_1kahler_potential_524789kahler_potential_524791kahler_potential_524793kahler_potential_524795kahler_potential_524797*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8В *0
f+R)
'__inference_restored_function_body_25722*
(kahler_potential/StatefulPartitionedCall░
IdentityIdentity1kahler_potential/StatefulPartitionedCall:output:0)^kahler_potential/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*:
_input_shapes)
':         :::::2T
(kahler_potential/StatefulPartitionedCall(kahler_potential/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
ў
╖
?__forward_dense_1_layer_call_and_return_conditional_losses_3862
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	мd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0*+
_input_shapes
:         м:*m
backward_function_nameSQ__inference___backward_dense_1_layer_call_and_return_conditional_losses_3839_3863:P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
Ї
╖
?__forward_dense_3_layer_call_and_return_conditional_losses_2630
inputs_0"
matmul_readvariableop_resource
identity

matmul
matmul_readvariableop

inputsИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:dd*
dtype02
MatMul/ReadVariableOpu
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0"
inputsinputs_0"
matmulMatMul:product:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0**
_input_shapes
:         d:*m
backward_function_nameSQ__inference___backward_dense_3_layer_call_and_return_conditional_losses_2616_2631:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
ы
Д
A__inference_dense_1_layer_call_and_return_conditional_losses_1766

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	мd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMul^
SquareSquareMatMul:product:0*
T0*'
_output_shapes
:         d2
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*+
_input_shapes
:         м::P L
(
_output_shapes
:         м
 
_user_specified_nameinputs
№
В
?__inference_dense_layer_call_and_return_conditional_losses_2150

inputs"
matmul_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	м*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
MatMul_
SquareSquareMatMul:product:0*
T0*(
_output_shapes
:         м2
Square_
IdentityIdentity
Square:y:0*
T0*(
_output_shapes
:         м2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :                  ::X T
0
_output_shapes
:                  
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
;
input_10
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:эl
└
biholomorphic

layer1

layer2

layer3

layer4

layer5

signatures
#_self_saveable_object_factories
	regularization_losses

	variables
trainable_variables
	keras_api
*S&call_and_return_all_conditional_losses
T_default_save_signature
U__call__"Р
_tf_keras_modelЎ{"class_name": "KahlerPotential", "name": "kahler_potential", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "KahlerPotential"}}
щ
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"╡
_tf_keras_layerЫ{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
╚
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"Н
_tf_keras_layerє{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"П
_tf_keras_layerї{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
w
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*\&call_and_return_all_conditional_losses
]__call__"П
_tf_keras_layerї{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
$w
#%_self_saveable_object_factories
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*^&call_and_return_all_conditional_losses
___call__"П
_tf_keras_layerї{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
╩
*w
#+_self_saveable_object_factories
,regularization_losses
-	variables
.trainable_variables
/	keras_api
*`&call_and_return_all_conditional_losses
a__call__"П
_tf_keras_layerї{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
,
bserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
╩
	regularization_losses
0metrics

	variables
1non_trainable_variables

2layers
3layer_regularization_losses
trainable_variables
4layer_metrics
U__call__
T_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
regularization_losses
5metrics
	variables
6non_trainable_variables

7layers
8layer_regularization_losses
trainable_variables
9layer_metrics
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	м2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
н
regularization_losses
:metrics
	variables
;non_trainable_variables

<layers
=layer_regularization_losses
trainable_variables
>layer_metrics
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
:	мd2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
н
regularization_losses
?metrics
	variables
@non_trainable_variables

Alayers
Blayer_regularization_losses
trainable_variables
Clayer_metrics
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
н
 regularization_losses
Dmetrics
!	variables
Enon_trainable_variables

Flayers
Glayer_regularization_losses
"trainable_variables
Hlayer_metrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
:dd2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
н
&regularization_losses
Imetrics
'	variables
Jnon_trainable_variables

Klayers
Llayer_regularization_losses
(trainable_variables
Mlayer_metrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
:d2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
н
,regularization_losses
Nmetrics
-	variables
Onon_trainable_variables

Players
Qlayer_regularization_losses
.trainable_variables
Rlayer_metrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
О2Л
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294╝
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
▀2▄
!__inference__wrapped_model_524801╢
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
є2Ё
/__inference_kahler_potential_layer_call_fn_2304╝
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_1         
ч2ф
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1893Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╠2╔
,__inference_biholomorphic_layer_call_fn_2111Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀2▄
?__inference_dense_layer_call_and_return_conditional_losses_1933Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴
$__inference_dense_layer_call_fn_2156Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
A__inference_dense_1_layer_call_and_return_conditional_losses_1917Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
&__inference_dense_1_layer_call_fn_1772Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
A__inference_dense_2_layer_call_and_return_conditional_losses_1925Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
&__inference_dense_2_layer_call_fn_2170Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
A__inference_dense_3_layer_call_and_return_conditional_losses_1840Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
&__inference_dense_3_layer_call_fn_1970Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
с2▐
A__inference_dense_4_layer_call_and_return_conditional_losses_1948Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
&__inference_dense_4_layer_call_fn_2340Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
3B1
$__inference_signature_wrapper_524818input_1У
!__inference__wrapped_model_524801n$*0в-
&в#
!К
input_1         
к "3к0
.
output_1"К
output_1         м
G__inference_biholomorphic_layer_call_and_return_conditional_losses_1893a/в,
%в"
 К
inputs         
к ".в+
$К!
0                  
Ъ Д
,__inference_biholomorphic_layer_call_fn_2111T/в,
%в"
 К
inputs         
к "!К                  б
A__inference_dense_1_layer_call_and_return_conditional_losses_1917\0в-
&в#
!К
inputs         м
к "%в"
К
0         d
Ъ y
&__inference_dense_1_layer_call_fn_1772O0в-
&в#
!К
inputs         м
к "К         dа
A__inference_dense_2_layer_call_and_return_conditional_losses_1925[/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ x
&__inference_dense_2_layer_call_fn_2170N/в,
%в"
 К
inputs         d
к "К         dа
A__inference_dense_3_layer_call_and_return_conditional_losses_1840[$/в,
%в"
 К
inputs         d
к "%в"
К
0         d
Ъ x
&__inference_dense_3_layer_call_fn_1970N$/в,
%в"
 К
inputs         d
к "К         dа
A__inference_dense_4_layer_call_and_return_conditional_losses_1948[*/в,
%в"
 К
inputs         d
к "%в"
К
0         
Ъ x
&__inference_dense_4_layer_call_fn_2340N*/в,
%в"
 К
inputs         d
к "К         и
?__inference_dense_layer_call_and_return_conditional_losses_1933e8в5
.в+
)К&
inputs                  
к "&в#
К
0         м
Ъ А
$__inference_dense_layer_call_fn_2156X8в5
.в+
)К&
inputs                  
к "К         мо
J__inference_kahler_potential_layer_call_and_return_conditional_losses_2294`$*0в-
&в#
!К
input_1         
к "%в"
К
0         
Ъ Ж
/__inference_kahler_potential_layer_call_fn_2304S$*0в-
&в#
!К
input_1         
к "К         б
$__inference_signature_wrapper_524818y$*;в8
в 
1к.
,
input_1!К
input_1         "3к0
.
output_1"К
output_1         