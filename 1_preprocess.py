import os
from multiprocessing import freeze_support
from unstructured_ingest.connector.local import SimpleLocalConfig
from unstructured_ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured_ingest.runner import LocalRunner
import nltk
nltk.download('averaged_perceptron_tagger')


def main():
    input_path = "/Users/govin/Projects/sprintdotnext/legal-advisor/documents/html"
    output_path = "/Users/govin/Projects/sprintdotnext/legal-advisor/local-ingest-output"

    runner = LocalRunner(
        processor_config=ProcessorConfig(
            # logs verbosity
            verbose=False,
            # the local directory to store outputs
            output_dir=output_path,
            num_processes=2,
        ),
        read_config=ReadConfig(),
        partition_config=PartitionConfig(
            partition_by_api=False,
        ),
        connector_config=SimpleLocalConfig(
            input_path=input_path,
            # whether to get the documents recursively from given directory
            recursive=False,
        ),
    )
    runner.run()

if __name__ == '__main__':
#    freeze_support()  # Optional: use for Windows/macOS compatibility with multiprocessing
    main()

