import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job_config", type=str, required=True)
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()

    # Read in YAML file
    yaml_path = os.path.join(".", "job_config", args.job_config)
    with open(yaml_path, "r") as f:
        job_config = yaml.safe_load(f)

    print(job_config)
    snippet_path = job_config['snippet_path']
    output_path = job_config['output_path']
    year = job_config['year']
    archive = job_config['archive']
    run_batches = job_config['run_batches']
    column_model_list = job_config['column_model_list']

    sub_name = args.job_config.split('.')[0]

    for column_dict in column_model_list:
        for column_name in column_dict:
            model_list = column_dict[column_name]
            input_path = os.path.join(snippet_path, column_name)
            os.makedirs(output_path, exist_ok=True)

            for model in model_list:
                model_output_name = model + '_' + column_name + '.csv'
                model_output_path = os.path.join(output_path, str(year), sub_name, column_name,
                                                 model, model_output_name)
                command = "sbatch run_batch.sh " + " ".join((input_path, model, model_output_path))
                os.system(command)


if __name__ == "__main__":
    main()
