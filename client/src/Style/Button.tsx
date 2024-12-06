import React from 'react';

interface ButtonProps {
    children: React.ReactNode;
    onClick?: () => void;
    className?: string;
}

export default function Button({ children, onClick, className }: ButtonProps) {
  return (
    <button onClick={onClick} className={`px-4 py-2 rounded-full ${className}`}>
      {children}
    </button>
  );
}