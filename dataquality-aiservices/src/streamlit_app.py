import streamlit as st
import pandas as pd
import global_controller as gc

if "data_type" not in st.session_state:
    st.session_state.data_type = None
if "DataFrame" not in st.session_state:
    st.session_state.DataFrame = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "file_uploder_obj" not in st.session_state:
    st.session_state.file_uploder_obj = None
if "Encoded_Data_Frame" not in st.session_state:
    st.session_state.Encoded_Data_Frame = None
if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = {}

DATA_TYPES = [None, "Time_Series_Data", "Tabular_Data",
              "t1_page", "t2_page", "t3_page",
              "ts1_page", "ts2_page"] #TODO new page: neue seitenname einbinden in liste
DATA_TYPES_SELECTBOX = ["Time_Series_Data", "Tabular_Data"]

def start():
    #st.write(st.session_state.Metadata) #TODO idee für Metadatenspeicherung, reicht list of list?
    st.header("Upload a Dataset", divider="rainbow")
    uploaded_file = st.file_uploader("",type=["csv"])
    if uploaded_file is not None:
        st.info (f"Your uploaded dataset:", icon="ℹ️")
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        gc.add_metadata("uploaded_file",gc.calculate_dataframe_metadata(df))
        gc.add_metadata("uploaded DataFrame",df.to_dict())
        #st.write(gc.get_all_metadata())
        st.info (f"To start the demonstrator please select a dataset type:", icon="ℹ️")
        dtype = st.selectbox("Which dataset type do you have uploaded?",
                             DATA_TYPES_SELECTBOX,
                             placeholder="Choose the type of the dataset",
                             index= None
                             )

        if st.button("Go on"):
            st.session_state.file_uploder_obj = uploaded_file
            if dtype == "Time_Series_Data":
                st.session_state.data_type = "ts1_page"
            else:
                st.session_state.data_type = dtype
            st.session_state.DataFrame = df
            st.session_state.filename = uploaded_file.name
            if dtype != None:
                gc.add_metadata ("Type of Data", dtype)
            st.rerun()


def back():
    st.session_state.data_type = None
    st.session_state.DataFrame = None
    st.session_state.filename = None
    if "metadata_store" in st.session_state:
        st.session_state.metadata_store = {}
    st.rerun()

dtype = st.session_state.data_type

home_page = st.Page(back, title="Upload another file", icon=":material/upload:")
tabular_page = st.Page(
    "tabular_data.py",
    title="Tabular Data",
    icon=":material/bar_chart:",
    default=(dtype == "Tabular_Data")
)
timeseries_page = st.Page(
    "time_series_data.py",
    title="Timseries Data",
    icon=":material/monitoring:",
    default=(dtype == "Time_Series_Data")
)
t1_page = st.Page( #TODO new page: anlegen und Titel und bild vergeben
    page="demonstrators/streamlit_demonstrator/streamlit_demonstrator.py",
    title="Demonstrator1",
    icon=":material/bar_chart:",
    default=(dtype == "t1_page")
)
t2_page = st.Page(
    page="demonstrators/anomaly_detection_tabular_data.py/anomaly_detection_tabular_data.py",
    title="Demonstrator2",
    icon=":material/bar_chart:",
    default=(dtype == "t2_page")
)
t3_page = st.Page(
    page="demonstrators/meanmode.py/meanmode.py",
    title="Demonstrator3",
    icon=":material/bar_chart:",
    default=(dtype == "t3_page")
)

ts1_page = st.Page(
    page="demonstrators/streamlit_demonstrator/streamlit_demonstrator.py",
    title="Demonstrator1",
    icon=":material/monitoring:",
    default=(dtype == "ts1_page")
)
ts2_page = st.Page(
    page="demonstrators/ANOMALYDETECTIONHMTEAM/ANOMALYDETECTIONHMTEAM/Anomalydetection.py",
    title="Anomalydetection",
    icon=":material/monitoring:",
    default=(dtype == "ts2_page")
)
summarising_page = st.Page(
    page="summarising_page.py",
    title="Summary",
    icon=":material/monitoring:",
    default=(dtype == "summarising_page")
)

#set section names and default page
home_pages = [home_page]
tabular_pages = [tabular_page]
timeseries_pages = [ts1_page] #[timeseries_page]
summary_pages = [summarising_page]

if st.session_state.data_type == None:
    #st.title("Welcome to the demonstrator")
    pass
#st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png") #hier kann man noch images einbinden KI Alianz logo?

page_dict = {}
#TODO hier weiter ausbauben um die gewünschte verstecken und weiter anzeige zu bekommen eventuell mit neuem session state in tabular zum entscheiden etc?
if st.session_state.data_type != None:
    page_dict["Start"] = home_pages

#Tabular Page handling
if st.session_state.data_type in ["Tabular_Data"]:
    page_dict["Tabular"] = tabular_pages
if st.session_state.data_type == "t1_page":  #TODO new page: session state data_type abfragen und wenn neue seite verfügbar, dann append und aufrufen
    #st.write(tabular_page._default)
    tabular_pages.append(t1_page)
    page_dict["Tabular"] = tabular_pages
if st.session_state.data_type == "t2_page":
    tabular_pages.append(t1_page)
    tabular_pages.append(t2_page)
    page_dict["Tabular"] = tabular_pages
if st.session_state.data_type == "t3_page":
    tabular_pages.append(t1_page)
    tabular_pages.append(t2_page)
    tabular_pages.append(t3_page)
    page_dict["Tabular"] = tabular_pages

#Timeseries Page handling
if st.session_state.data_type in ["Time_Series_Data", "ts1_page"]:
    page_dict["Time_Series"] = timeseries_pages
if st.session_state.data_type in ["ts2_page"]:
    timeseries_pages.append(ts2_page)
    page_dict["Time_Series"] = timeseries_pages

# summary einbinden in den reiter
if st.session_state.data_type != None:
    page_dict["Summary"] = summary_pages


if len(page_dict) > 0:
    pg = st.navigation(page_dict)
else:
    pg = st.navigation([st.Page(start)])

pg.run()
