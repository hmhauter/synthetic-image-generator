import React from 'react';
import { Snackbar, IconButton } from '@mui/material';
import { Alert } from '@mui/material';
import { Close } from '@mui/icons-material';

export type ToastType = 'info' | 'warning' | 'error';

interface ToastProps {
  open: boolean;
  type: ToastType;
  message: string;
  description?: string;
  onClose: () => void;
}

const Toast = ({ open, type, message, description, onClose }: ToastProps) => {
  return (
    <Snackbar open={open} autoHideDuration={4000} onClose={onClose}>
      <Alert severity={type} variant="filled" action={
        <IconButton size="small" color="inherit" onClick={onClose}>
          <Close fontSize="small" />
        </IconButton>
      }>
        {description && (
          <React.Fragment>
            {description}
            <br></br>
          </React.Fragment>
        )}
        {message}
      </Alert>
    </Snackbar>
  );
};

export default Toast;