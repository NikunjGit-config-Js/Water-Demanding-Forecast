# Interrupted Phase 9 run

This attempt was stopped during LSTM recursive inference after TensorFlow could
not initialize CUDA and repeated `Model.predict` calls rebuilt input pipelines
for each forecast step. Partial model files are retained as audit evidence and
must not be treated as completed Phase 9 results. The inference path was changed
to an equivalent direct eager model call before the final rerun.
