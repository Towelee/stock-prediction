	'1��;@'1��;@!'1��;@	���;w�?���;w�?!���;w�?"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'�'1��;@���K;@A�O��n�?Y{�G�z�?*	     �I@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�� �rh�?!������@@)����Mb�?1______?@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�~j�t��?!������7@)�~j�t��?1������7@:Preprocessing2F
Iterator::Model��~j�t�?!������B@)�I+��?1������5@:Preprocessing2S
Iterator::Model::ParallelMap����Mb�?!______/@)����Mb�?1______/@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!������@)�~j�t�h?1������@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!______�?)����MbP?1______�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B97.7 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���K;@���K;@!���K;@      ��!       "      ��!       *      ��!       2	�O��n�?�O��n�?!�O��n�?:      ��!       B      ��!       J	{�G�z�?{�G�z�?!{�G�z�?R      ��!       Z	{�G�z�?{�G�z�?!{�G�z�?JCPU_ONLY