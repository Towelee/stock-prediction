	+���/P@+���/P@!+���/P@	�4���?�4���?!�4���?"h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'�+���/P@u�V~O@A�G�z��?Y���Q��?*	      D@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat���Q��?!     �B@)y�&1��?1     �A@:Preprocessing2F
Iterator::Model�I+��?!    �;@)�I+��?1    �;@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�I+��?!    �;@)�I+��?1    �;@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!      @)�~j�t�h?1      @:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����MbP?!      @)����MbP?1      @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B97.3 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	u�V~O@u�V~O@!u�V~O@      ��!       "      ��!       *      ��!       2	�G�z��?�G�z��?!�G�z��?:      ��!       B      ��!       J	���Q��?���Q��?!���Q��?R      ��!       Z	���Q��?���Q��?!���Q��?JCPU_ONLY