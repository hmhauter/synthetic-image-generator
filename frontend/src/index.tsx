import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import Homepage from 'Homepage';
import UploadBackground from 'UploadBackground';
import UploadForeground from 'UploadForeground';


const router = createBrowserRouter([
  {
    path: "/",
    element: <Homepage />,
  },
  {
    path: "/generateImages",
    element: <App />,
  },
  {
    path: "/uploadBackground",
    element: <UploadBackground />,
  },
  {
    path: "/uploadForeground",
    element: <UploadForeground />,
  }
]);

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);



   root.render(
    <React.StrictMode>
      <RouterProvider router={router} />
    </React.StrictMode>
  );






