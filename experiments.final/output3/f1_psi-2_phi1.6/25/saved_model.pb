ци
—£
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
Њ
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
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878оѓ
l
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_name
Variable
e
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes

:*
dtype0

NoOpNoOp
ї
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ц
valueмBй Bв
Б
biholomorphic

layer1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
	regularization_losses

trainable_variables
	keras_api
Y
w
	variables
regularization_losses
trainable_variables
	keras_api

0
 

0
≠
metrics
	variables
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
layer_metrics
trainable_variables
 
 
 
 
≠
metrics
	variables

layers
	regularization_losses
layer_regularization_losses
non_trainable_variables
layer_metrics

trainable_variables
A?
VARIABLE_VALUEVariable#layer1/w/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
≠
metrics
	variables

layers
regularization_losses
layer_regularization_losses
non_trainable_variables
layer_metrics
trainable_variables
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
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ѕ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_409950
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
љ
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_410048
§
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_410061 Ь
±
J
.__inference_biholomorphic_layer_call_fn_410008

inputs
identity”
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_biholomorphic_layer_call_and_return_conditional_losses_4099042
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ƒ
q
*__inference_zerolayer_layer_call_fn_409941
input_1
unknown
identityИҐStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_zerolayer_layer_call_and_return_conditional_losses_4099332
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ъ
k
$__inference_signature_wrapper_409950
input_1
unknown
identityИҐStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_4098472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ9
e
I__inference_biholomorphic_layer_call_and_return_conditional_losses_409904

inputs
identityE
ConjConjinputs*'
_output_shapes
:€€€€€€€€€2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
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
€€€€€€€€€2
MatrixBandPart/num_upper∆
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:€€€€€€€€€2
RealO
ImagImagReshape:output:0*'
_output_shapes
:€€€€€€€€€2
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
:€€€€€€€€€22
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
:2€€€€€€€€€2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2
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
#boolean_mask/Prod/reduction_indicesҐ
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
$boolean_mask/strided_slice_1/stack_2Є
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
$boolean_mask/strided_slice_2/stack_2Є
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
boolean_mask/concat/axisь
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisы
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
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
:€€€€€€€€€€€€€€€€€€2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
™
О
K__inference_width_one_dense_layer_call_and_return_conditional_losses_410015

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€::X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я
v
0__inference_width_one_dense_layer_call_fn_410022

inputs
unknown
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_width_one_dense_layer_call_and_return_conditional_losses_4099192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ю
Х
__inference__traced_save_410048
file_prefix'
#savev2_variable_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
value3B1 B+_temp_f133d12d24754b5d9efcb3bfe22034b5/part2	
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename√
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesМ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
SaveV2/shape_and_slicesа
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*!
_input_shapes
: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::

_output_shapes
: 
‘
Ђ
E__inference_zerolayer_layer_call_and_return_conditional_losses_409933
input_1
width_one_dense_409928
identityИҐ'width_one_dense/StatefulPartitionedCallр
biholomorphic/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_biholomorphic_layer_call_and_return_conditional_losses_4099042
biholomorphic/PartitionedCallј
'width_one_dense/StatefulPartitionedCallStatefulPartitionedCall&biholomorphic/PartitionedCall:output:0width_one_dense_409928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_width_one_dense_layer_call_and_return_conditional_losses_4099192)
'width_one_dense/StatefulPartitionedCallu
LogLog0width_one_dense/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
LogЕ
IdentityIdentityLog:y:0(^width_one_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€:2R
'width_one_dense/StatefulPartitionedCall'width_one_dense/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
й

z
"__inference__traced_restore_410061
file_prefix
assignvariableop_variable

identity_2ИҐAssignVariableOp…
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*V
valueMBKB#layer1/w/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesТ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B 2
RestoreV2/shape_and_slicesµ
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

IdentityШ
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
™
О
K__inference_width_one_dense_layer_call_and_return_conditional_losses_409919

inputs"
matmul_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMuld
IdentityIdentityMatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :€€€€€€€€€€€€€€€€€€::X T
0
_output_shapes
:€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ќ9
e
I__inference_biholomorphic_layer_call_and_return_conditional_losses_410003

inputs
identityE
ConjConjinputs*'
_output_shapes
:€€€€€€€€€2
ConjШ
einsum/EinsumEinsuminputsConj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
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
€€€€€€€€€2
MatrixBandPart/num_upper∆
MatrixBandPartMatrixBandParteinsum/Einsum:output:0!MatrixBandPart/num_lower:output:0!MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
MatrixBandParto
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Reshape/shape~
ReshapeReshapeMatrixBandPart:band:0Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
ReshapeO
RealRealReshape:output:0*'
_output_shapes
:€€€€€€€€€2
RealO
ImagImagReshape:output:0*'
_output_shapes
:€€€€€€€€€2
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
:€€€€€€€€€22
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
:2€€€€€€€€€2
	transposeR
AbsAbstranspose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2
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
#boolean_mask/Prod/reduction_indicesҐ
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
$boolean_mask/strided_slice_1/stack_2Є
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
$boolean_mask/strided_slice_2/stack_2Є
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
boolean_mask/concat/axisь
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concatЦ
boolean_mask/ReshapeReshapetranspose:y:0boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2
boolean_mask/ReshapeП
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
boolean_mask/Reshape_1/shapeЩ
boolean_mask/Reshape_1ReshapeSqueeze:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:22
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€2
boolean_mask/WhereШ
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axisы
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
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
:€€€€€€€€€€€€€€€€€€2
transpose_1l
IdentityIdentitytranspose_1:y:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
пZ

!__inference__wrapped_model_409847
input_1<
8zerolayer_width_one_dense_matmul_readvariableop_resource
identityИv
zerolayer/biholomorphic/ConjConjinput_1*'
_output_shapes
:€€€€€€€€€2
zerolayer/biholomorphic/Conjб
%zerolayer/biholomorphic/einsum/EinsumEinsuminput_1%zerolayer/biholomorphic/Conj:output:0*
N*
T0*+
_output_shapes
:€€€€€€€€€*
equation
ai,aj->aij2'
%zerolayer/biholomorphic/einsum/Einsum¶
0zerolayer/biholomorphic/MatrixBandPart/num_lowerConst*
_output_shapes
: *
dtype0	*
value	B	 R 22
0zerolayer/biholomorphic/MatrixBandPart/num_lowerѓ
0zerolayer/biholomorphic/MatrixBandPart/num_upperConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€22
0zerolayer/biholomorphic/MatrixBandPart/num_upperЊ
&zerolayer/biholomorphic/MatrixBandPartMatrixBandPart.zerolayer/biholomorphic/einsum/Einsum:output:09zerolayer/biholomorphic/MatrixBandPart/num_lower:output:09zerolayer/biholomorphic/MatrixBandPart/num_upper:output:0*
T0*+
_output_shapes
:€€€€€€€€€2(
&zerolayer/biholomorphic/MatrixBandPartЯ
%zerolayer/biholomorphic/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2'
%zerolayer/biholomorphic/Reshape/shapeё
zerolayer/biholomorphic/ReshapeReshape-zerolayer/biholomorphic/MatrixBandPart:band:0.zerolayer/biholomorphic/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€2!
zerolayer/biholomorphic/ReshapeЧ
zerolayer/biholomorphic/RealReal(zerolayer/biholomorphic/Reshape:output:0*'
_output_shapes
:€€€€€€€€€2
zerolayer/biholomorphic/RealЧ
zerolayer/biholomorphic/ImagImag(zerolayer/biholomorphic/Reshape:output:0*'
_output_shapes
:€€€€€€€€€2
zerolayer/biholomorphic/ImagМ
#zerolayer/biholomorphic/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2%
#zerolayer/biholomorphic/concat/axisГ
zerolayer/biholomorphic/concatConcatV2%zerolayer/biholomorphic/Real:output:0%zerolayer/biholomorphic/Imag:output:0,zerolayer/biholomorphic/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€22 
zerolayer/biholomorphic/concat°
&zerolayer/biholomorphic/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&zerolayer/biholomorphic/transpose/permя
!zerolayer/biholomorphic/transpose	Transpose'zerolayer/biholomorphic/concat:output:0/zerolayer/biholomorphic/transpose/perm:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2#
!zerolayer/biholomorphic/transposeЪ
zerolayer/biholomorphic/AbsAbs%zerolayer/biholomorphic/transpose:y:0*
T0*'
_output_shapes
:2€€€€€€€€€2
zerolayer/biholomorphic/Abs†
-zerolayer/biholomorphic/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-zerolayer/biholomorphic/Sum/reduction_indicesњ
zerolayer/biholomorphic/SumSumzerolayer/biholomorphic/Abs:y:06zerolayer/biholomorphic/Sum/reduction_indices:output:0*
T0*
_output_shapes
:22
zerolayer/biholomorphic/SumЕ
zerolayer/biholomorphic/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2 
zerolayer/biholomorphic/Less/yЄ
zerolayer/biholomorphic/LessLess$zerolayer/biholomorphic/Sum:output:0'zerolayer/biholomorphic/Less/y:output:0*
T0*
_output_shapes
:22
zerolayer/biholomorphic/LessФ
"zerolayer/biholomorphic/LogicalNot
LogicalNot zerolayer/biholomorphic/Less:z:0*
_output_shapes
:22$
"zerolayer/biholomorphic/LogicalNotЪ
zerolayer/biholomorphic/SqueezeSqueeze&zerolayer/biholomorphic/LogicalNot:y:0*
T0
*
_output_shapes
:22!
zerolayer/biholomorphic/Squeeze≠
*zerolayer/biholomorphic/boolean_mask/ShapeShape%zerolayer/biholomorphic/transpose:y:0*
T0*
_output_shapes
:2,
*zerolayer/biholomorphic/boolean_mask/ShapeЊ
8zerolayer/biholomorphic/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8zerolayer/biholomorphic/boolean_mask/strided_slice/stack¬
:zerolayer/biholomorphic/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:zerolayer/biholomorphic/boolean_mask/strided_slice/stack_1¬
:zerolayer/biholomorphic/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:zerolayer/biholomorphic/boolean_mask/strided_slice/stack_2ђ
2zerolayer/biholomorphic/boolean_mask/strided_sliceStridedSlice3zerolayer/biholomorphic/boolean_mask/Shape:output:0Azerolayer/biholomorphic/boolean_mask/strided_slice/stack:output:0Czerolayer/biholomorphic/boolean_mask/strided_slice/stack_1:output:0Czerolayer/biholomorphic/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:24
2zerolayer/biholomorphic/boolean_mask/strided_sliceƒ
;zerolayer/biholomorphic/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2=
;zerolayer/biholomorphic/boolean_mask/Prod/reduction_indicesВ
)zerolayer/biholomorphic/boolean_mask/ProdProd;zerolayer/biholomorphic/boolean_mask/strided_slice:output:0Dzerolayer/biholomorphic/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2+
)zerolayer/biholomorphic/boolean_mask/Prod±
,zerolayer/biholomorphic/boolean_mask/Shape_1Shape%zerolayer/biholomorphic/transpose:y:0*
T0*
_output_shapes
:2.
,zerolayer/biholomorphic/boolean_mask/Shape_1¬
:zerolayer/biholomorphic/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:zerolayer/biholomorphic/boolean_mask/strided_slice_1/stack∆
<zerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<zerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_1∆
<zerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<zerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_2»
4zerolayer/biholomorphic/boolean_mask/strided_slice_1StridedSlice5zerolayer/biholomorphic/boolean_mask/Shape_1:output:0Czerolayer/biholomorphic/boolean_mask/strided_slice_1/stack:output:0Ezerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_1:output:0Ezerolayer/biholomorphic/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask26
4zerolayer/biholomorphic/boolean_mask/strided_slice_1±
,zerolayer/biholomorphic/boolean_mask/Shape_2Shape%zerolayer/biholomorphic/transpose:y:0*
T0*
_output_shapes
:2.
,zerolayer/biholomorphic/boolean_mask/Shape_2¬
:zerolayer/biholomorphic/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:zerolayer/biholomorphic/boolean_mask/strided_slice_2/stack∆
<zerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2>
<zerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_1∆
<zerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<zerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_2»
4zerolayer/biholomorphic/boolean_mask/strided_slice_2StridedSlice5zerolayer/biholomorphic/boolean_mask/Shape_2:output:0Czerolayer/biholomorphic/boolean_mask/strided_slice_2/stack:output:0Ezerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_1:output:0Ezerolayer/biholomorphic/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask26
4zerolayer/biholomorphic/boolean_mask/strided_slice_2÷
4zerolayer/biholomorphic/boolean_mask/concat/values_1Pack2zerolayer/biholomorphic/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:26
4zerolayer/biholomorphic/boolean_mask/concat/values_1¶
0zerolayer/biholomorphic/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0zerolayer/biholomorphic/boolean_mask/concat/axisМ
+zerolayer/biholomorphic/boolean_mask/concatConcatV2=zerolayer/biholomorphic/boolean_mask/strided_slice_1:output:0=zerolayer/biholomorphic/boolean_mask/concat/values_1:output:0=zerolayer/biholomorphic/boolean_mask/strided_slice_2:output:09zerolayer/biholomorphic/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+zerolayer/biholomorphic/boolean_mask/concatц
,zerolayer/biholomorphic/boolean_mask/ReshapeReshape%zerolayer/biholomorphic/transpose:y:04zerolayer/biholomorphic/boolean_mask/concat:output:0*
T0*'
_output_shapes
:2€€€€€€€€€2.
,zerolayer/biholomorphic/boolean_mask/Reshapeњ
4zerolayer/biholomorphic/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€26
4zerolayer/biholomorphic/boolean_mask/Reshape_1/shapeщ
.zerolayer/biholomorphic/boolean_mask/Reshape_1Reshape(zerolayer/biholomorphic/Squeeze:output:0=zerolayer/biholomorphic/boolean_mask/Reshape_1/shape:output:0*
T0
*
_output_shapes
:220
.zerolayer/biholomorphic/boolean_mask/Reshape_1√
*zerolayer/biholomorphic/boolean_mask/WhereWhere7zerolayer/biholomorphic/boolean_mask/Reshape_1:output:0*'
_output_shapes
:€€€€€€€€€2,
*zerolayer/biholomorphic/boolean_mask/Whereа
,zerolayer/biholomorphic/boolean_mask/SqueezeSqueeze2zerolayer/biholomorphic/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:€€€€€€€€€*
squeeze_dims
2.
,zerolayer/biholomorphic/boolean_mask/Squeeze™
2zerolayer/biholomorphic/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2zerolayer/biholomorphic/boolean_mask/GatherV2/axisу
-zerolayer/biholomorphic/boolean_mask/GatherV2GatherV25zerolayer/biholomorphic/boolean_mask/Reshape:output:05zerolayer/biholomorphic/boolean_mask/Squeeze:output:0;zerolayer/biholomorphic/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2/
-zerolayer/biholomorphic/boolean_mask/GatherV2•
(zerolayer/biholomorphic/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(zerolayer/biholomorphic/transpose_1/permэ
#zerolayer/biholomorphic/transpose_1	Transpose6zerolayer/biholomorphic/boolean_mask/GatherV2:output:01zerolayer/biholomorphic/transpose_1/perm:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2%
#zerolayer/biholomorphic/transpose_1џ
/zerolayer/width_one_dense/MatMul/ReadVariableOpReadVariableOp8zerolayer_width_one_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/zerolayer/width_one_dense/MatMul/ReadVariableOpв
 zerolayer/width_one_dense/MatMulMatMul'zerolayer/biholomorphic/transpose_1:y:07zerolayer/width_one_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 zerolayer/width_one_dense/MatMulГ
zerolayer/LogLog*zerolayer/width_one_dense/MatMul:product:0*
T0*'
_output_shapes
:€€€€€€€€€2
zerolayer/Loge
IdentityIdentityzerolayer/Log:y:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0**
_input_shapes
:€€€€€€€€€::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
;
input_10
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:–1
Ў
biholomorphic

layer1
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 __call__
!_default_save_signature
*"&call_and_return_all_conditional_losses"э
_tf_keras_modelг{"class_name": "zerolayer", "name": "zerolayer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "zerolayer"}}
ƒ
	variables
	regularization_losses

trainable_variables
	keras_api
#__call__
*$&call_and_return_all_conditional_losses"µ
_tf_keras_layerЫ{"class_name": "Biholomorphic", "name": "biholomorphic", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "biholomorphic", "trainable": true, "dtype": "float32"}}
µ
w
	variables
regularization_losses
trainable_variables
	keras_api
%__call__
*&&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "WidthOneDense", "name": "width_one_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 
metrics
	variables
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
layer_metrics
trainable_variables
 __call__
!_default_save_signature
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
,
'serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
metrics
	variables

layers
	regularization_losses
layer_regularization_losses
non_trainable_variables
layer_metrics

trainable_variables
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
:2Variable
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
≠
metrics
	variables

layers
regularization_losses
layer_regularization_losses
non_trainable_variables
layer_metrics
trainable_variables
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ш2х
*__inference_zerolayer_layer_call_fn_409941∆
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
я2№
!__inference__wrapped_model_409847ґ
Л≤З
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
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
У2Р
E__inference_zerolayer_layer_call_and_return_conditional_losses_409933∆
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
Ў2’
.__inference_biholomorphic_layer_call_fn_410008Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *
 
у2р
I__inference_biholomorphic_layer_call_and_return_conditional_losses_410003Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *
 
Џ2„
0__inference_width_one_dense_layer_call_fn_410022Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *
 
х2т
K__inference_width_one_dense_layer_call_and_return_conditional_losses_410015Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
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
annotations™ *
 
3B1
$__inference_signature_wrapper_409950input_1П
!__inference__wrapped_model_409847j0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€Ѓ
I__inference_biholomorphic_layer_call_and_return_conditional_losses_410003a/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€€€€€€€€€€
Ъ Ж
.__inference_biholomorphic_layer_call_fn_410008T/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К€€€€€€€€€€€€€€€€€€Э
$__inference_signature_wrapper_409950u;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€≥
K__inference_width_one_dense_layer_call_and_return_conditional_losses_410015d8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ Л
0__inference_width_one_dense_layer_call_fn_410022W8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€€€€€€€€€€
™ "К€€€€€€€€€•
E__inference_zerolayer_layer_call_and_return_conditional_losses_409933\0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
*__inference_zerolayer_layer_call_fn_409941O0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "К€€€€€€€€€