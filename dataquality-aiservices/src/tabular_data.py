import streamlit as st
import global_controller as gc
import pandas as pd

st.header("Target Variable and Problem Type", divider="rainbow")
#st.write(f"You are logged in as {st.session_state.data_type}")


file = st.session_state.file_uploder_obj #st.file_uploader("Select your data", type='csv')  #Typen prüfen oder erweitern
if file is not None:
    column_list = []
    file.seek(0)
    df =pd.read_csv(file)
    st.info (f"Your uploaded dataset:", icon="ℹ️")
    st.write (df)
    st.subheader ("Select a target variable to determ a problem type ", divider="rainbow")
    column_list += df.columns.tolist()
    target_variable = st.selectbox("Which column do you want to peform as the target variable?",
                                   column_list,
                                   placeholder="Select the target variable",
                                   index=None)

    if target_variable != None:
        #st.write(df[target_variable])
        gc.add_metadata("Target variable",target_variable)
        problem_type=gc.infer_problem_type(df[target_variable])
        #st.write(f'For the selected target variable, {target_variable} is the '
        #         f'recomanded problem type: {problem_type}. To continue with that'
        #         f' problem type click on continue. \nIf you want to change the '
        #         f'problem type, please click the change')
        st.info (f"A heuristic determined **{problem_type}**  as the problem type:", icon="ℹ️")
        #st.subheader(f"Result of the heuristic is: {problem_type}", divider="rainbow")


        problem_type_list = gc.get_problem_type_list() # hier nicht ganz optimal
        #st.write (problem_type_list)
        new_list = [problem_type]
        for type in problem_type_list:
            if type != problem_type:
                new_list.append(type)

        problem_type = st.selectbox("Do you want to change the Problem Type?",new_list, index=0)
        gc.add_metadata ("Heuristic outcome", new_list[0])
        if problem_type == "regression":
            if st.button("Go on"):
                gc.add_metadata("User has chosen", problem_type)
                st.session_state.Encoded_Data_Frame = gc.one_hot_encode_df(df)
                st.session_state.data_type = "t1_page"
                st.rerun()
        elif problem_type in ["multiclass", "binary"]:
            if st.button("Go on"):
                gc.add_metadata("User has chosen", problem_type)
                st.session_state.Encoded_Data_Frame = gc.one_hot_encode_df (df)
                st.session_state.data_type = "t1_page"
                st.rerun()

else:
    #Button to return to the start page
    st.write("Error, please exit and restart the program.")




