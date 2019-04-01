
# coding: utf-8

# ## Type abbreviations

# The code and docs sometimes use *type abbreviations* to avoid type signatures getting unwieldy. Here's a list of all abbreviations for composite types for convenient access.

# ## From [`core`](/core.html#core)

# - `AnnealFunc` = `Callable`\[\[`Number`,`Number`,`float`], `Number`]
# - `ArgStar` = `Collection`\[`Any`]
# - `BatchSamples` = `Collection`\[`Tuple`\[`Collection`\[`int`], `int`]]
# - `Classes` = `Collection`\[`Any`]
# - `DataFrameOrChunks` = `Union[DataFrame, pd.io.parsers.TextFileReader]`
# - `FilePathList` = `Collection`\[`Path`]
# - `Floats` = `Union`\[`float`, `Collection`\[`float`]]
# - `ImgLabels` = `Collection`\[`ImgLabel`]
# - `KeyFunc` = `Callable`\[\[`int`], `int`]
# - `KWArgs` = `Dict`\[`str`,`Any`]
# - `ListOrItem` = `Union`\[`Collection`\[`Any`],`int`,`float`,`str`]
# - `ListRules` = `Collection`\[`Callable`\[\[`str`],`str`]]
# - `ListSizes` = `Collection`\[`Tuple`\[`int`,`int`]]
# - `NPArrayableList` = `Collection`\[`Union`\[`np`.`ndarray`, `list`]]
# - `NPArrayList` = `Collection`\[`np`.`ndarray`]
# - `OptDataFrame` = `Optional`\[`DataFrame`]
# - `OptListOrItem` = `Optional`\[`ListOrItem`]
# - `OptRange` = `Optional`\[`Tuple`\[`float`,`float`]]
# - `OptStrTuple` = `Optional`\[`Tuple`\[`str`,`str`]]
# - `OptStats` = `Optional`\[`Tuple`\[`np`.`ndarray`, `np`.`ndarray`]]
# - `PathOrStr` = `Union`\[`Path`,`str`]
# - `PBar` = `Union`\[`MasterBar`, `ProgressBar`]
# - `Point`=`Tuple`\[`float`,`float`]
# - `Points`=`Collection`\[`Point`]
# - `Sizes` = `List`\[`List`\[`int`]]
# - `SplitArrayList` = `List`\[`Tuple`\[`np`.`ndarray`,`np`.`ndarray`]]
# - `StartOptEnd`=`Union`\[`float`,`Tuple`\[`float`,`float`]]
# - `StrList` = `Collection`\[`str`]
# - `Tokens` = `Collection`\[`Collection`\[`str`]]
# - `OptStrList` = `Optional`\[`StrList`]

# ## From [`torch_core`](/torch_core.html#torch_core)

# - `BoolOrTensor` = `Union`\[`bool`,`Tensor`]
# - `FloatOrTensor` = `Union`\[`float`,`Tensor`]
# - `IntOrTensor` = `Union`\[`int`,`Tensor`]
# - `ItemsList` = `Collection`\[`Union`\[`Tensor`,[`ItemBase`](/core.html#ItemBase),'`ItemsList`',`float`,`int`]]
# - `LambdaFunc` = `Callable`\[\[`Tensor`],`Tensor`]
# - `LayerFunc` = `Callable`\[ [[`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)],`None`]
# - [`Model`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) = [`nn`](https://pytorch.org/docs/stable/nn.html#torch-nn).`Module`
# - `ModuleList` = `Collection`\[[`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)]
# - `OptOptimizer` = `Optional`\[[`optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)]
# - `ParamList` = `Collection`\[[`nn.Parameter`](https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter)]
# - `Rank0Tensor` = `NewType`('`OneEltTensor`', `Tensor`)
# - `SplitFunc` = `Callable`\[[`Model`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)], `List`[`Model`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)]]
# - `SplitFuncOrIdxList` = `Union`\[`Callable`, `Collection`\[`ModuleList`]]
# - `TensorOrNumber` = `Union`\[`Tensor`,`Number`]
# - `TensorOrNumList` = `Collection`\[`TensorOrNumber`]
# - `TensorImageSize` = `Tuple`\[`int`,`int`,`int`]
# - `Tensors` = `Union`\[`Tensor`, `Collection`\['`Tensors`']]
# - `Weights` = `Dict`\[`str`,`Tensor`]
# - `AffineFunc` = `Callable`\[\[`KWArgs`], [`AffineMatrix`](https://pytorch.org/docs/stable/tensors.html#torch-tensor)]
# - `HookFunc` = `Callable`\[[`Model`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module), `Tensors`, `Tensors`], `Any`]
# - `LogitTensorImage` = `TensorImage`
# - `LossFunction` = `Callable`\[\[`Tensor`, `Tensor`], `Rank0Tensor`]
# - `MetricFunc` = `Callable`\[\[`Tensor`,`Tensor`],`TensorOrNumber`]
# - `MetricFuncList` = `Collection`\[`MetricFunc`]
# - `MetricsList` = `Collection`\[`TensorOrNumber`]
# - `OptLossFunc` = `Optional`\[`LossFunction`]
# - `OptMetrics` = `Optional`\[`MetricsList`]
# - `OptSplitFunc` = `Optional`\[`SplitFunc`]
# - `PixelFunc` = `Callable`\[\[`TensorImage`, `ArgStar`, `KWArgs`], `TensorImage`]
# - `CoordFunc` = `Callable`\[[`FlowField`](/vision.image.html#FlowField), `TensorImageSize`, `ArgStar`, `KWArgs`], `LogitTensorImage`]
# - `LightingFunc` = `Callable`\[\[`LogitTensorImage`, `ArgStar`, `KWArgs`], `LogitTensorImage`]
