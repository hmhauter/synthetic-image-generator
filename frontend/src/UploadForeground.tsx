import React, { useRef, useState } from 'react';
import { Link } from "react-router-dom";
import { Box, Button, FormControl, Grid, InputLabel, ListItemText, MenuItem, Select, SelectChangeEvent, TextField } from "@mui/material";
import Banner from "Banner";
import Toast, { ToastType } from "Toast";
import { postSegment } from 'api/api';


interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function Foreground() {

  const [image, setImage] = useState<File | null>(null);
  const [imageRect, setImageRect] = useState<DOMRect | null>(null);

  const [classType, setClassType] = useState<string>('Add Class');
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
  const [isDrawing, setIsDrawing] = useState<Boolean>(false);
  const [availableClasses, setAvailableClasses] = useState<string[]>(["Add Class"]);
  const [customInput, setCustomInput] = useState('');

  // TOAST STATES
  const [toastOpen, setToastOpen] = useState(false);
  const [toastType, setToastType] = useState<ToastType>('info');
  const [toastMessage, setToastMessage] = useState('');
  const [toastDescription, setToastDescription] = useState('');


  // const imageRef = useRef<HTMLImageElement>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  React.useEffect(() => {
    document.body.style.overflow = 'hidden';

    const handleGlobalMouseMove = (event: any) => {
      // Your global mouse move logic
      if (imageRect && boundingBox && isDrawing == true) {
        const currentX = event.clientX;
        const currentY = event.clientY;
        if (currentX >= imageRect.left && currentX <= imageRect.right && currentY >= imageRect.top && currentY <= imageRect.bottom) {
          const startX = boundingBox.x;
          const startY = boundingBox.y;
          const newBoundingBox: BoundingBox = {
            x: Math.min(startX, currentX),
            y: Math.min(startY, currentY),
            width: Math.abs(startX - currentX),
            height: Math.abs(startY - currentY),
          };

          setBoundingBox(newBoundingBox);
        }
      }

    };

    const handleGlobalMouseDown = (event: any) => {
      var imageRect_temp = null
      const { clientX, clientY } = event;
      var imageRef = document.getElementById('foreground-image');
      var rect = null
      if (imageRef) {
        rect = imageRef.getBoundingClientRect();
        if (rect) {
          setImageRect(rect)
          imageRect_temp = rect
        }
      }
      if (imageRect_temp && clientX >= imageRect_temp.left && clientX <= imageRect_temp.right && clientY >= imageRect_temp.top && clientY <= imageRect_temp.bottom) {
        setIsDrawing(true);
        setBoundingBox({ x: clientX, y: clientY, width: 0, height: 0 });
      }
    };

    const handleGlobalMouseUp = () => {
      setIsDrawing(false);
    };


    // Attach global event listeners
    document.addEventListener('mousemove', handleGlobalMouseMove);
    document.addEventListener('mousedown', handleGlobalMouseDown);
    document.addEventListener('mouseup', handleGlobalMouseUp);

    return () => {
      document.body.style.overflow = 'visible';
      document.removeEventListener('mousemove', handleGlobalMouseMove);
      document.removeEventListener('mousedown', handleGlobalMouseDown);
      document.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [isDrawing, boundingBox, image, imageRect]);


  const handleImageInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setImage(event.target.files[0]);
      setBoundingBox(null)
    }
  };

  const handleClassTypeChange = (event: SelectChangeEvent<typeof classType>) => {
    const {
      target: { value },
    } = event;
    setClassType(
      value,
    );
  };


  const handleClearBoundingBox = () => {
    dragStartRef.current = null;
    setBoundingBox(null)
    setToastType('info')
    setToastMessage("Bounding Box was cleared successfully")
    setToastOpen(true)
  };

  const handleSubmit = () => {
    // Submit bounding box logic goes here
    setToastOpen(true)

  };

  const convertBase64ToFile = (base64String: string, filename = 'image.png') => {
    // Convert base64 to Blob
    const byteCharacters = atob(base64String);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });

    // Create File object from Blob
    const file = new File([blob], filename, { type: 'image/png' });

    return file
  };

  const handleSegment = async () => {
    // Submit bounding box logic goes here
    setToastType('info')
    setToastMessage("Successfully submitted image for segmentation")
    setToastOpen(true)

    var base64_image = await getBase64(1024,1024)
    console.log(base64_image)

    if(image && boundingBox && imageRect) {
      var imageRef = document.getElementById('foreground-image')! as HTMLImageElement;
      var bbox = [boundingBox.x-imageRect.left, boundingBox.y-imageRect.top, boundingBox.width, boundingBox.height]
      var size = [imageRef.width, imageRef.height]

      postSegment({
        image: base64_image,
        size: JSON.stringify(size),
        bbox: JSON.stringify(bbox)
      })
      .then(response => {
        console.log(response.data.message)
        console.log(response.data.image)
        var segmented_img =convertBase64ToFile(response.data.image)
        console.log("YESSSSSSSSSSSS")
        setImage(segmented_img)
        setBoundingBox(null)
      
      })
      .catch(error => console.log(error))

    }

  };

  function getBase64(maxWidth: number, maxHeight: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
  
      reader.onload = function () {
        const img = new Image();
        img.src = reader.result as string;
  
        img.onload = function () {
          const canvas = document.createElement('canvas');
          let width = img.width;
          let height = img.height;
  
          // Calculate new dimensions while maintaining aspect ratio
          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }
  
          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }
  
          canvas.width = width;
          canvas.height = height;
  
          const ctx = canvas.getContext('2d');
          ctx?.drawImage(img, 0, 0, width, height);
  
          // Get the resized image as base64
          const resizedBase64 = canvas.toDataURL('image/jpeg'); // You can change the format if needed
  
          resolve(resizedBase64);
        };
      };
  
      reader.onerror = (error) => {
        reject(error);
      };
  
      reader.readAsDataURL(image!);
    });
  }
    
  

  

  const handleCloseToast = () => {
    setToastOpen(false);
  };


  const handleClassTChange = (event: any) => {
    setClassType(event.target.value);
  };

  const handleAddClass = () => {
    setAvailableClasses(prevArray => [...prevArray, customInput])
  }

  const handleCustomInputChange = (event: any) => {
    setCustomInput(event.target.value);
  };

  return (
    <div>
      <Banner />
      <Box sx={{ m: 3 }}>
        <Grid container spacing={3}>
          {/* Column 1: Select Object Button and Image */}
          <Grid item xs={20} md={7}>

            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <label htmlFor="foreground-input">
                <Button variant="contained" component="span" sx={{
                  backgroundColor: "#001965",
                  color: "#FCF2F5",
                  borderColor: "#282c34",
                  ":hover": {
                    backgroundColor: "#637099",
                    color: "white",
                  },
                  m: 1,
                }}>
                  Select Object
                </Button>
              </label>
            </Box>
            <input type="file" id="foreground-input" accept="image/*" onChange={handleImageInputChange} />
            {image && (
              <Box sx={{ mt: 2 }}>
                <div>
                  <img
                    id="foreground-image"
                    src={URL.createObjectURL(image)}
                    alt="Object"
                    style={{ maxHeight: "400px" }}
                    draggable="false"
                  />
                  {boundingBox && (
                    <div
                      style={{
                        position: 'absolute',
                        border: '2px solid red',
                        left: boundingBox.x,
                        top: boundingBox.y,
                        width: boundingBox.width,
                        height: boundingBox.height,
                      }}
                    />
                  )}
                </div>
              </Box>
            )}


          </Grid>
          {/* Column 2: Buttons to Select Class Type, Clear BB, and Segment */}
          <Grid item xs={12} md={3}>
            <Box sx={{ display: 'flex', alignItems: 'center', flexDirection: 'column', justifyContent: 'flex-start' }}>

              <FormControl fullWidth sx={{ minWidth: 80, m: 1, mt: 2 }}>
                <InputLabel>Class Type</InputLabel>
                <Select label="Class Type" value={classType} onChange={handleClassTChange} sx={{ minWidth: 80 }}>
                  {availableClasses?.map((name) => (
                    <MenuItem key={name} value={name}>
                      <ListItemText primary={name} />
                    </MenuItem>
                  ))}
                </Select>

                {classType === 'Add Class' && (
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} md={6}>
                      <TextField
                        label="Add Class"
                        value={customInput}
                        onChange={handleCustomInputChange}
                        fullWidth  
                        sx={{ mt: 3 }}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Button
                        onClick={handleAddClass}
                        variant="contained"
                        sx={{
                          backgroundColor: "#001965",
                          color: "#FCF2F5",
                          borderColor: "#282c34",
                          mt: 3,
                          ml: 1,
                          flexGrow: 1, // Allow the Button to take up available space
                          ":hover": {
                            backgroundColor: "#637099",
                            color: "white",
                          },
                        }}
                      >
                        Add
                      </Button>
                    </Grid>
                  </Grid>
                )}
              </FormControl>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start' }}>

                <Button variant="contained" onClick={handleClearBoundingBox} sx={{
                  backgroundColor: "#001965",
                  color: "#FCF2F5",
                  borderColor: "#282c34",
                  ":hover": {
                    backgroundColor: "#637099",
                    color: "white",
                  },
                  m: 1,
                }}>Clear BB</Button>
                <Button variant="contained" 
                onClick={handleSegment}
                sx={{
                  backgroundColor: "#001965",
                  color: "#FCF2F5",
                  borderColor: "#282c34",
                  ":hover": {
                    backgroundColor: "#637099",
                    color: "white",
                  },
                  m: 1,
                }}>Segment</Button>

              </Box>
            </Box>
          </Grid>
          {/* Column 3: Submit and Back Buttons */}
          <Grid item xs={12} md={2}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Button
                  variant="outlined"
                  onClick={handleSubmit}
                  sx={{
                    backgroundColor: "#001965",
                    color: "#FCF2F5",
                    borderColor: "#282c34",
                    ":hover": {
                      backgroundColor: "#637099",
                      color: "white",
                    },
                    m: 1,
                  }}
                >
                  Submit
                </Button>
                <Link to="/">
                  <Button sx={{
                    backgroundColor: "#001965",
                    color: "#FCF2F5",
                    borderColor: "#282c34",
                    ":hover": {
                      backgroundColor: "#637099",
                      color: "white",
                    },
                    m: 1,
                    mt: 2,
                  }}>
                    Back
                  </Button>
                </Link>
              </Box>


            </Box>
          </Grid>
        </Grid>
      </Box>
      <Toast open={toastOpen} type={toastType} message={toastMessage} description={toastDescription} onClose={handleCloseToast} />

    </div>
  );
};





