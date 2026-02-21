import streamlit as st
import pandas as pd
import random
import sys
from io import StringIO
import re
import io
import torch
import sortinghat.pylib as pl
import global_controller as gc


# make streamlit whole pagewidth
#st.set_page_config(layout='wide')


RandomForest_classes = ['numeric', 'categorical', 'datetime',
                        'sentence', 'url', 'embedded-number',
                        'list', 'not-generalizable', 'context-specific'
                        ]








def read_file(uploaded_file, developer_mode):
    try:
        if uploaded_file.name.endswith('.csv'):
            file_content = uploaded_file.getvalue().decode('utf-8')
            # remove comments from file
            file_content, nr_removed_comments = re.subn(r'(?m)^ *#.*\n?', '', file_content)
            if nr_removed_comments > 0 and developer_mode:
                st.info(f"Removed {nr_removed_comments} comments from file.", icon="ℹ️")
            # remove empty lines
            file_content, nr_removed_lines = re.subn(r'^$\n', '', file_content, flags=re.MULTILINE)
            if nr_removed_lines > 0 and developer_mode:
                st.info(f"Removed {nr_removed_lines} empty lines from file.", icon="ℹ️")
            # Convert String into StringIO
            csvStringIO = StringIO(file_content)
            # first row of the dataframe in csv file
            first_row = file_content.split('\n')[0].split(",")
            # check if first row contains only strings
            if all(isinstance(item, str) for item in first_row):
                # if all strings, use them as column names
                if developer_mode:
                    st.info('CSV file contains a header. Columns are set.', icon="ℹ️")
                df = pd.read_csv(csvStringIO)
            else:
                # if not, csv file has no header
                if developer_mode:
                    st.info('CSV file contains no header.', icon="ℹ️")
                df = pd.read_csv(csvStringIO, header=None)
            # check if index column existing
            if (df[df.columns[0]].values == range(len(df))).all():
                # if yes, set it as index
                if developer_mode:
                    st.info('There was an index column. Index is set.', icon="ℹ️")
                df.set_index(df.columns[0], inplace=True)        
        if uploaded_file.name.endswith('.parquet'):
            # add preprocessing for parquet files
            df = pd.read_parquet(uploaded_file)
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            # add preprocessing for excel files
            excel_file = pd.ExcelFile(uploaded_file)
            df = excel_file.parse(excel_file.sheet_names[0])
    except Exception as e:
        st.error(f'There was an error parsing the file: {e}', icon="🚨")
        sys.exit(1)
    #st.header('Overview of the working dataset', divider='rainbow')
    st.info (f"Your uploaded dataset:", icon="ℹ️")
    #st.write("Your uploaded data:")
    st.write(df)
    return df


def get_features(df, n_values=10):
    features = []
    for c in df.columns:
        nr_distinct_values = len(df[c].unique())

        # check if given number of values works
        if n_values >= nr_distinct_values:
            n_values = nr_distinct_values
        
        # select up to n random unique values
        distinct_values = random.sample(df[c].unique().tolist(), n_values)

        # remove None values
        distinct_values = [q for q in distinct_values if q is not None]
        distinct_values = [q for q in distinct_values if str(q).lower() != 'none']

        # remove nan values
        distinct_values = [q for q in distinct_values if str(q).lower() != 'nan']
        
        # remove potential duplicates
        distinct_values = list(set(distinct_values))

        # sort the list
        distinct_values = sorted(distinct_values)


        # insert each feature and its values to the list
        f = {
            'feature': c,
            'distinct_values': str(sorted(distinct_values)),
            'nr_distinct_values': nr_distinct_values
        }
        features.append(f)

    return features


def make_predictions(df, model):
    df = df*1 # turn true and false entries to 0 and 1 (required for RF)
    dataFeaturized = pl.FeaturizeFile(df)
    columns_of_interest = dataFeaturized.columns[:15]
    result = dataFeaturized.filter(columns_of_interest)
    result.insert(loc=0, column='prediction', value="") # predictions as first column



    if model == 'RandomForest':
        dataFeaturized1 = pl.FeatureExtraction(dataFeaturized)
        # new columns are int, make all strings to escape error from model.fit
        dataFeaturized1.columns = dataFeaturized1.columns.astype(str)
        y_RF = pl.Load_RF(dataFeaturized1.values)
        result['prediction'] = y_RF
        # map numbers back to classes
        for i in range(len(result['prediction'])):
            result.loc[i, 'prediction'] = RandomForest_classes[int(result['prediction'][i])]
    
    return result




def get_piveau(uploaded_file, predicted_df):
    results_metadata = ""
    piveau_str = ""
    for i in range(len(predicted_df)):
        results_metadata += f"{predicted_df['Attribute_name'].values[i]} ({predicted_df['prediction'].values[i]})"
        if not i == len(predicted_df)-1:
            results_metadata += ', '
    if uploaded_file.name.endswith('.csv'):
        piveau_str = \
            f'@prefix dcat: <http://www.w3.org/ns/dcat#>\n\n' \
            f'<url>\n' \
            f'\tdcat:Distribution ;\n\n' \
            f'\tdcat:mediaType <http://www.iana.org/assignments/media-types/text/csv> ;\n\n' \
            f'\tdct:title "{uploaded_file.name} dataset + feature scale identification" ;\n\n' \
            f'\tdct:description "{results_metadata}".'
    if uploaded_file.name.endswith('.xlsx'):
        piveau_str = \
            f'```\n@prefix dcat: <http://www.w3.org/ns/dcat#>\n\n' \
            f'<url>\n' \
            f'\tdcat:Distribution ;\n\n' \
            f'\tdcat:mediaType <http://publications.europa.eu/resource/authority/file-type/XLS> ;\n\n' \
            f'\tdct:title "{uploaded_file.name} dataset + feature scale identification" ;\n\n' \
            f'\tdct:description "{results_metadata}".\n```'
    if uploaded_file.name.endswith('.parquet'):
        piveau_str = \
            f'```\n@prefix dcat: <http://www.w3.org/ns/dcat#>\n\n' \
            f'<url>\n' \
            f'\tdcat:Distribution ;\n\n' \
            f'\tdcat:mediaType <https://www.iana.org/assignments/media-types/application/vnd.apache.parquet> ;\n\n' \
            f'\tdct:title "{uploaded_file.name} dataset + feature scale identification" ;\n\n' \
            f'\tdct:description "{results_metadata}".\n```'
    return piveau_str
    


@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

@st.cache_data
def convert_df_to_parquet(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_parquet()

@st.cache_data
def convert_df_to_excel(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        df.to_excel(writer)
    return output.getvalue()



def create_download(uploaded_file, piveau_entry, data):
    st.header('Download results', divider='rainbow')
    if uploaded_file.name.endswith('.csv'):
        st.download_button(
            label="Download data",
            data=convert_df_to_csv(data),
            file_name=uploaded_file.name,
            mime="text/csv",
        )
    if uploaded_file.name.endswith('.parquet'):
        st.download_button(
            label="Download data",
            data=convert_df_to_parquet(data),
            file_name=uploaded_file.name,
        )
    if uploaded_file.name.endswith('.xlsx'):
        st.download_button(
            label="Download data",
            data=convert_df_to_excel(data),
            file_name=uploaded_file.name,
        )

    st.download_button(
        label="Download piveau entry",
        data=piveau_entry,
        file_name=str(uploaded_file.name).split('.')[0] + '.ttl', # remove .csv, .parquet and .xlsx
    )



def display_piveau_results(uploaded_file, predicted_df, df):
    piveau_entry = get_piveau(uploaded_file, predicted_df)
    st.header('Created piveau entry', divider='rainbow')
    st.write(f"```\n{piveau_entry}\n```")

    create_download(uploaded_file, piveau_entry, data=df)




@st.fragment
def user_edit_results(predicted_df, df, model, uploaded_file):
    st.subheader('Infered Feature Types', divider='rainbow')
    st.info(f'You are able to modify the results by changing the values in' \
            f' the column **edit_result**:', icon="ℹ️")# TODO einbinden in andere
    
    if 'edit_result' not in predicted_df.columns:
        predicted_df.insert(loc=1, column='edit_result', value=predicted_df['prediction'])
    gc.add_metadata("Infered Feature Types",predicted_df.to_dict())

    with st.form("user_edit_results"):

        if model == 'RandomForest':
            predicted_df = st.data_editor(
                predicted_df,
                column_config={
                    "edit_result": st.column_config.SelectboxColumn(
                        "edit_result",
                        help="You can change the prediction of the model in this column.",
                        options=RandomForest_classes,
                        required=True
                    )
                },
                use_container_width=True,
                disabled=[x for x in df.columns if not x == 'edit_result']
            )
        
        submitted = st.form_submit_button("Apply changes")

    if submitted:

        predicted_df['prediction'] = predicted_df['edit_result']
        predicted_df.drop(columns=['edit_result'], inplace=True)
        st.success('Successfully applied the changes!', icon="✅")
        gc.add_metadata ("Infered Feature Types changed by user",predicted_df.to_dict())
        #display_piveau_results(uploaded_file, predicted_df, df)

        st.info (f"You are able to go to the next page. ", icon="ℹ️")
        if st.session_state.data_type in ["ts1_page", " Time_Series_Data"]:
            st.session_state.data_type = "ts2_page"
            st.rerun ()
        elif st.session_state.data_type in ["t1_page", "Tabular_Data"]:
            st.session_state.data_type = "t2_page"
            st.rerun ()

    



def main():

    st.info(f"If you want to skip this part of the demonstrator please click on the button below:", icon="ℹ️")
    _, right = st.columns([5, 1])  # Hack to be on the right side
    if right.button("Skip Page"):
        if st.session_state.data_type in ["ts1_page", " Time_Series_Data"]:
            st.session_state.data_type = "ts2_page"
            st.rerun ()
        elif st.session_state.data_type in ["t1_page", "Tabular_Data"]:
            st.session_state.data_type = "t2_page"
            st.rerun ()

    st.header ("Demonstrator llm in the forest", divider="rainbow")
    developer_mode = False #st.toggle("Activate developer mode")
    uploaded_file = st.session_state.file_uploder_obj #st.file_uploader("Select a dataset", type=(["csv","parquet","xlsx", "xls"]))

    if uploaded_file is not None:
        df = read_file(uploaded_file, developer_mode)
        st.info (f"Select a algorithm:", icon="ℹ️")
        model = st.selectbox(
            "What model you want to use for prediction?",
            placeholder="Choose an option", #TODO einbinden in andere Dateien
            options=("RandomForest"),
            index=None
        )

        if st.button("Run"):
            gc.add_metadata ("User has chosen this model", model)
            if not model:
                st.error(f'Please select a model from the dropdown menu!', icon="🚨")
            else:
                if developer_mode:
                    st.info(f"Selected {model} as model for predictions.", icon="ℹ️")
                
                if model == 'RandomForest':
                    if developer_mode:
                        st.info(f"Will use the following 9 classes for predictions: \n" \
                                f"{', '.join(RandomForest_classes)}", icon="ℹ️")
                    with st.spinner('Operation in progress. Please wait.'):
                        try:
                            predicted_df = make_predictions(df, model)
                            user_edit_results(predicted_df, df, model, uploaded_file)
                        except Exception as e:
                            st.error(f'There was an error predicting the file: {e}', icon="🚨")
                            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
                            sys.exit(1)




if __name__ == '__main__':
    main()

main()