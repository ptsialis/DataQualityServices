import React from 'react';
import { useTranslation } from 'react-i18next';

function UploadModal({ onFileSelect, onClose }) {
  const { t } = useTranslation();

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{t('uploadTitle')}</h2>
          <button onClick={onClose} className="close-button">&times;</button>
        </div>
        <div
          className="dropzone"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <p>{t('dragDrop')}</p>
          <p className="file-limit">{t('fileLimit')}</p>
          <label htmlFor="file-input" className="browse-button">
            {t('browseFiles')}
          </label>
          <input
            id="file-input"
            type="file"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </div>
      </div>
    </div>
  );
}

export default UploadModal;