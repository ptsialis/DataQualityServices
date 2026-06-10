import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: {
    translation: {
      piveau: "Piveau",
      preprocessing: "Image Preprocessing",
      upload: "Upload Data",
      title: "Make Your Dataset Machine-Learning Ready!",
      uploaded: "Uploaded",
      uploadTitle: "Upload your Data",
      dragDrop: "Drag and drop file here",
      fileLimit: "Limit 200MB per file • CSV",
      browseFiles: "Browse files",
      original: "Dataset",
      inference: "Feature Type Inference",
      imputation: "Imputation of Data",
      anomaly: "Anomaly Detected",
      personal: "Personalize Data Detection",
      summary: "Summary",
      metadata: "Create Meta-Data",
      cleaned: "Show cleaned Data",
      model: "Model",
      imageprep: "Image Preprocessing",
      downloads: "Downloads",
      dataUploaded: "Data uploaded",
      selectTargetVariable: "Select the target variable",
      submit: "Submit",
      detected: "detected",
      rawData: "Show Raw Data",
      seeStatistics: "Show Statistics of Original Data",
      seeGraphs: "See Graphs of Data",
      seeFeatureResults: "See Feature Typ Inference Results",
      seeExplanation: "See Explanation Feature Type Inference",
      imputationResults: "Imputation Results",
      seeExplanationImp: "See Explanation",
      anomalyResults: "Anomaly Detection",
      seeExplanationAno: "See Explanation Anomaly",
      personalizedDetect: "Personalize Data Detection",
      showSummary: "Show Summary",
      showMeta: "Show Meta Data",
      MetaImpu: "See Meta Data Imputation",
      boxplot: " Boxplot of Numerical Features",
      toShow: 'show Rows',
      all: 'all',
      datagraphs: "Data Graphs",
      boxplots: 'Boxplots',
      histograms: 'Histograms',
      correlationMatrix: 'Correlation Matrix',
      featureImportance: 'Feature Importance'


    }
  },
  de: {
    translation: {
      piveau: "Piveau",
      preprocessing: "Bildvorverarbeitung",
      upload: "Daten Hochladen",
      title: "Machen Sie Ihren Datensatz Machine-Learning-Ready!",
      uploaded: "Hochgeladen",
      uploadTitle: "Daten hochladen",
      dragDrop: "Datei hierher ziehen und ablegen",
      fileLimit: "Max. 200MB pro Datei • CSV",
      browseFiles: "Dateien durchsuchen",
      original: "Datensatz",
      inference: "Merkmalstyp Erkennung",
      imputation: "Imputation von Daten",
      anomaly: "Erkennung von Anomalien",
      personal: "Erkennung von persönlichen Informationen",
      summary: "Zusammenfassung",
      metadata: "Meta-Daten erstellen",
      cleaned: "Gesäuberte Daten anzeigen",
      model: "Model",
      imageprep: "Bildvorverarbeitung",
      downloads: "Downloads",
      dataUploaded: "Daten hochgeladen",
      selectTargetVariable: "Wählen Sie die Zielvariable aus",
      submit: "Submit",
      detected: "erkannt",
      rawData: "Rohdaten anzeigen",
      seeStatistics: "Statistiken der Originaldaten anzeigen",
      seeGraphs: "Siehe Grafiken der Daten",
      seeFeatureResults: "Ergebnisse der Merkmalstyp Erkennung",
      seeExplanation: "Erklärung der Merkmalstyp Erkennung",
      imputationResults: "Imputation Ergebnisse",
      seeExplanationImp: "Erklärung anzeigen",
      anomalyResults: "Anomalien Ergebnisse",
      seeExplanationAno: "Erklärung Anomalien anzeigen",
      personalizedDetect: "Erkennung von persönlichen Informationen",
      showSummary: "Zusammenfassung anzeigen",
      showMeta: "Meta Daten anzeigen",
      MetaImpu: "Meta Daten Imputation anzeigen",
      boxplot: " Boxplot",
      toShow: 'Zeilen anzeigen',
      all: 'alle',
      datagraphs: "Daten-Grafiken",
      boxplots: "Boxplots",
      histograms: "Histogramme",
      correlationMatrix: "Korrelationsmatrix",
      featureImportance: "Feature Importance"

    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: "en",
    fallbackLng: "en",
    interpolation: { escapeValue: false }
  });

export default i18n;
