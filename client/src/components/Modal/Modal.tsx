import React from 'react';

interface ModalProps {
  isOpen: boolean;
  closeModal: () => void;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, closeModal, children }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-md w-full">
        <button
          className="absolute top-2 right-2 text-xl"
          onClick={closeModal}
        >
          &times;
        </button>
        {children}
      </div>
    </div>
  );
};

export default Modal;
