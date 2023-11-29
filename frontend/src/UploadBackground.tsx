import React, { useState } from 'react';
import { Box, Button, Grid, Typography } from '@mui/material';
import Banner from 'Banner';
import { Link } from 'react-router-dom';

export default function Background() {

  const [image1, setImage1] = useState<File | undefined>(undefined);
  const [image2, setImage2] = useState<File | undefined>(undefined);

  const handleImage1Change = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImage1(file);
    }
  };

  const handleImage2Change = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImage2(file);
    }
  };

  const handleImage1Clear = () => {
    setImage1(undefined);
  };

  const handleImage2Clear = () => {
    setImage2(undefined);
  };


  const onSubmit = async (
    event: React.MouseEvent<HTMLButtonElement>
  ): Promise<void> => {
    console.log("Submit")
  }

  return (
    <div>
      <Banner></Banner>
      <Box sx={{ m: 3 }}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={5}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <label htmlFor="image1-input">
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
                  Select Background
                </Button>
              </label>
            </Box>
            <Grid item xs={20} md={5}>
              {image1 && (
                <Box sx={{ ml: 2 }}>
                  <img src={URL.createObjectURL(image1)} alt="Image 1" width={100} />
                  <Button variant="outlined" onClick={handleImage1Clear} sx={{
                    backgroundColor: "#001965",
                    color: "#FCF2F5",
                    borderColor: "#282c34",
                    ":hover": {
                      backgroundColor: "#637099",
                      color: "white",
                    },
                    m: 1,
                  }}>
                    Clear
                  </Button>
                </Box>
              )}
            </Grid>
            <input type="file" id="image1-input" accept="image/*" onChange={handleImage1Change} />
          </Grid>
          <Grid item xs={12} md={5}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <label htmlFor="image2-input">
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
                  Select Background ROI
                </Button>
              </label>
            </Box>
            <Grid item xs={20} md={5}>
              {image2 && (
                <Box sx={{ ml: 2 }}>
                  <img src={URL.createObjectURL(image2)} alt="Image 2" width={100} />
                  <Button variant="outlined" onClick={handleImage2Clear} sx={{
                    backgroundColor: "#001965",
                    color: "#FCF2F5",
                    borderColor: "#282c34",
                    ":hover": {
                      backgroundColor: "#637099",
                      color: "white",
                    },
                    m: 1,
                  }}>
                    Clear
                  </Button>
                </Box>
              )}
            </Grid>

            <input type="file" id="image2-input" accept="image/*" onChange={handleImage2Change} />
          </Grid>
          <Grid item xs={12} md={2}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>

              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <Button
                  variant="outlined"
                  onClick={onSubmit}
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
    </div>
  );
};