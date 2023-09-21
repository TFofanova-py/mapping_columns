import argparse
import pandas as pd
import re
from datetime import datetime as datetime
from chains import overall_chain


def get_column_description(ser: pd.Series, ser_name: str) -> str:
    description = ""
    examples_str = ", ".join([str(x) for x in ser.unique()[:2]])
    description += f"{ser_name}"
    description += f" (Examples: {examples_str})" if examples_str else ""
    return description


def extract_table_description(data: pd.DataFrame,
                              hide_column_names=False) -> str:
    description = ""

    for i, column in enumerate(data.columns):
        ser_name = i if hide_column_names else column
        description += "\t-" + get_column_description(data[column], ser_name) + "\n"

    return description


def create_column_dict(data: pd.DataFrame) -> dict:
    result_dict = {}

    for i, column in enumerate(data.columns):
        result_dict[i] = get_column_description(data[column], column)

    return result_dict


def parse_output(resp: str, verbose=False) -> dict:
    if not resp.endswith("\n"):
        resp += "\n"

    fields = re.findall(
        r"\"([^\"\n]*)\"\: *\"([^\"\n]*)\", *[\"\']?([^\n]*)[\"\']?,?\n",
        resp)

    mapping_dict = {}
    for col_name, map_col, transformation in fields:
        transformation = re.sub(r"[\"\',]$", "", transformation.strip())
        transformation = None if transformation == 'None' else transformation
        mapping_dict[col_name] = {"source_column": map_col, "transformation": transformation}

    if verbose:
        print(f"Mapping instructions:\n\n{mapping_dict}")

    return mapping_dict


def write_source(df_source: pd.DataFrame, df_template: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    mapped_df = pd.DataFrame(columns=df_template.columns)

    for col_name, map_instructions in mapping.items():
        try:
            mapped_df[col_name] = df_source[map_instructions["source_column"]]
            if map_instructions["transformation"] is not None:
                mapped_df[col_name] = mapped_df[col_name].apply(eval(map_instructions["transformation"]))
                mapped_df[col_name] = mapped_df[col_name].astype(df_template[col_name].dtypes.name)

        except Exception as e:
            print(e)
            continue

    return mapped_df


def prepare_inputs(data_source: pd.DataFrame, data_template: pd.DataFrame, verbose=False) -> dict:
    assert data_source.shape[0] > 0, "Source table is empty, that's no reason to match columns"
    # get table descriptions for making prompt
    source_description = extract_table_description(data_source)
    source_only_data_description = extract_table_description(data_source, hide_column_names=True)
    template_description = extract_table_description(data_template)

    source_colum_dict = create_column_dict(data_source)

    if verbose:
        print(f"""\nYou are making column mapping for:\n\nSource Table Columns:\n{source_description}\nTemplate Table Fields:\n{template_description}""")

    return {"source_without_column_names": source_only_data_description,
            "columns_dict": source_colum_dict,
            "source": source_description,
            "template": template_description}


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/table_B.csv")
    parser.add_argument("--template", type=str, default="data/template.csv")
    parser.add_argument("--target", type=str, default="data/target.csv")
    parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args()

    try:
        # source and template DataFrames
        df_source = pd.read_csv(args.source)
        df_template = pd.read_csv(args.template)
    except FileNotFoundError as e:
        print(e)

    else:
        verbose = not args.quiet
        # get inputs dictionary
        inputs = prepare_inputs(df_source, df_template, verbose=verbose)

        # get mapping instructions from LLM
        chain_result = overall_chain(inputs)

        # extract mapping instructions from response
        output_mapping = chain_result["mapping"]
        mapping = parse_output(output_mapping, verbose=verbose)

        # apply mapping
        df = write_source(df_source, df_template, mapping)
        assert df.shape[0] == df_source.shape[0] + df_template.shape[0], \
            "Something went wrong, shape of target don't match sum of shapes"
        df.to_csv(args.target, index=False)

        if verbose:
            print(f"\nTarget table is saved at {args.target}")
