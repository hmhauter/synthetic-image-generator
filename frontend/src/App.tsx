import React, { useEffect, useState } from "react";
import "./App.scss";
import { postGeneratorConfig } from "./api/api";

// load material ui component
import Alert from "@mui/material/Alert";
import Autocomplete from "@mui/material/Autocomplete";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import FormControl from "@mui/material/FormControl";
import TextField from "@mui/material/TextField";
import Snackbar from "@mui/material/Snackbar";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import CircularProgress from "@mui/material/CircularProgress";

// import api
import { ApiResponse, getText, postText } from "./api/api";
import { Input, InputLabel, ListItemText, MenuItem, OutlinedInput, Select, SelectChangeEvent, Slider, Switch } from "@mui/material";
import { Link } from "react-router-dom";

export interface expData {
  experimentData: String;
  cost: Number;
}

function App() {
  const [isOpenToast, setIsOpenToast] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isError, setIsError] = useState<boolean>(false);

  const [selectedClasses, setSelectedClasses] = useState<string[]>([]);
  const [splitValue, setSplitValue] = React.useState<number[]>([70, 85]);
  const [augmentValue, setAugmentValue] = React.useState<number>(0);
  const [outputFormat, setOutputFormat] = useState<String>("YOLO");
  const [numberImages, setNumberImages] = useState<String>("500");
  const [allowOverlapping, setAllowOverlapping] = useState<boolean>(true);


  const availableClasses = [
    "Pen",
    "Cap",
    "Cardbox"
  ]

  const formatOptions = [
    "YOLO",
    "COCO",
    "Pascal VOC"
  ]

  // const fetchExperimentData = async (): Promise<void> => {
  //   await getExperimentData().then((response) => {
  //     if (response.success)
  //       response.data
  //         ? setExperimentData(response.data)
  //         : setExperimentData("");
  //   });
  // };


  const getSliderText = () => {
    return "Train-Val-Test: " + String(splitValue[0]) + "-" + String(splitValue[1]-splitValue[0]) + "-" + String(100-splitValue[1])
  }

  // const onSubmit = async (
  //   event: React.MouseEvent<HTMLButtonElement>
  // ): Promise<void> => {
  //   console.log("HELP")
  //   event.preventDefault();
  //   setIsLoading(false); // TO DO
  //   console.log("Trying to find this gene:"+selectedGene)
  //   let _response = await postGene(selectedGene);
  //   if (_response.success) {
  //     console.log("POST WAS SUCCESSFULL");
  //     console.log(_response)

  //     setIsOpenToast(true);
  //     setIsLoading(false);
  //   }
  // };

  const onSubmit = async (
    event: React.MouseEvent<HTMLButtonElement>
  ): Promise<void> => {
    console.log("Submit")
    postGeneratorConfig({
      'classes': JSON.stringify(selectedClasses),
      'number': JSON.stringify(numberImages),
      'augmented': JSON.stringify(augmentValue),
      'isOverlapping': JSON.stringify(allowOverlapping),
      'split': JSON.stringify(splitValue),
      'output': JSON.stringify(formatOptions)
    })
    .then(response => {
      console.log(response)
    })
    .catch(error => console.log(error))
    // getText()
    // .then(response => console.log(response.data))
    // .catch(error => console.log(error))
    // postText("Data String to post")
    // .then(response => console.log(response))
    // .catch(error => console.log(error))
  }

  // async function get(): Promise<void> {
  //   let _response = await getExperimentData();
  //   if (_response.success) {
  //     setExperimentData(_response.data!);
  //   } else {
  //     setIsError(true);
  //   }
  // }


  const handleChangeSlider = (event: Event, newValue: number | number[]) => {
    setSplitValue(newValue as number[]);
  };

  const handleChangeAugmentedImg = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      setAugmentValue(newValue as number);
    }
  };

  const handleSwitchChange = () => {
    setAllowOverlapping((prev) => !prev);
  };


  function renderToast(severity: string): React.ReactNode {
    return (
      <Snackbar
        open={isOpenToast}
        autoHideDuration={8000}
        onClose={handleClose}
      >
        {severity == "success" ? (
          <Alert
            onClose={handleClose}
            severity="success"
            sx={{
              width: "100%",
              backgroundColor: "#282c34",
              color: "#F06434",
              borderColor: "#282c34",
            }}
          >
            Gene was searched successfully!
          </Alert>
        ) : (
          <Alert
            onClose={handleClose}
            severity="error"
            sx={{
              width: "100%",
              backgroundColor: "#282c34",
              color: "#F06434",
              borderColor: "#282c34",
            }}
          >
            Error! Please refresh the page and contact your administrator!
          </Alert>
        )}
      </Snackbar>
    );
  }

  
  const handleClassChange = (event: SelectChangeEvent<typeof selectedClasses>) => {
    const {
      target: { value },
    } = event;
    setSelectedClasses(
      typeof value === 'string' ? value.split(',') : value,
    );
  };

  const handleFormatChange = (event: SelectChangeEvent<typeof outputFormat>) => {
    const {
      target: { value },
    } = event;
    setOutputFormat(
      value,
    );
  };

  const handleNumImagesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setNumberImages(event.target.value);
  };


  const handleClose = (
    event: React.SyntheticEvent | Event,
    reason?: string
  ) => {
    if (reason === "clickaway") {
      return;
    }
    setIsOpenToast(false);
  };

  const backgroundImagesList = ["Pen", "Card"]

  const renderOption = (props: any, option: String, { selected }: { selected: boolean }) => (
    <li {...props}>
      <input type="checkbox" checked={selected} readOnly />
      {option}
    </li>
  );

  const calcNumberOfImages = () => {
    return Number(numberImages)*Number(augmentValue) + Number(numberImages)
  }

  return (
    <div className="App">
      <header className="App-header">Image Generator</header>
      <header className="App-header-small">Line Clearence Project</header>
      <div className="App-container">
        <Box className="App-shop-box">
        <FormControl sx={{ m: 1, width: 300 }}>
        <InputLabel id="demo-multiple-checkbox-label">Foreground Classes</InputLabel>
        <Select
          labelId="demo-multiple-checkbox-label"
          id="demo-multiple-checkbox"
          multiple
          value={selectedClasses}
          onChange={handleClassChange}
          input={<OutlinedInput label="Classes" />}
          renderValue={(selected) => selected.join(', ')}
        >
          {availableClasses.map((name) => (
            <MenuItem key={name} value={name}>
              <Checkbox checked={selectedClasses.indexOf(name) > -1} />
              <ListItemText primary={name} />
            </MenuItem>
          ))}
        </Select>
      </FormControl>
          </Box>
          </div>
          <div className="App-container">
          <Box className="App-container-textfield">
          <TextField
          margin="normal"
          label="Number of Images"
          variant="outlined"
          value={numberImages}
          onChange={handleNumImagesChange}
        ></TextField>
        </Box>
        </div>
        <div className="App-container">
          <Box className="App-container-textfield">
        <text className="App-container-label">Number of augmented images per generated image: </text>
        <Slider
          getAriaLabel={() => 'Number of augmented images'}
          value={augmentValue}
          onChange={handleChangeAugmentedImg}
          step={1}
          valueLabelDisplay="auto"
          marks min={0} max={3}
        />
        </Box>
      </div>
      <div className="App-container">
      <Box width={"50%"}>
      <text className="App-container-label">Allow overlapping objects: </text>
          <Switch
            checked={allowOverlapping}
            onChange={handleSwitchChange}
            color="primary" // You can customize the color
          />
      </Box>
      </div>
      <div className="App-container">
      <Box width={"50%"}>
        <text>{getSliderText()}</text>
        <Slider
        getAriaLabel={() => 'Test-Train-Val Split'}
        value={splitValue}
        onChange={handleChangeSlider}
        step={5}
        valueLabelDisplay="auto"
        getAriaValueText={getSliderText}
        valueLabelFormat={getSliderText}
        marks
        disableSwap
      />
        </Box>
      </div>
      <div className="App-container">
      <Box className="App-shop-box">
        <FormControl sx={{ m: 1, width: 300 }}>
          <InputLabel id="demo-simple-select-label">Output Format</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={outputFormat}
            label="Output Format"
            onChange={handleFormatChange}
          >
          {formatOptions.map((name) => (
            <MenuItem key={name} value={name}>
              <ListItemText primary={name} />
            </MenuItem>))}
          </Select>
        </FormControl>
        </Box>
      </div>
      <div>
        <header>Number of generated images in total: {calcNumberOfImages()}</header>
      </div>
      <Box>
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
          Generate Images
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
          }}>Back
          </Button>
          </Link>
          </Box>
      {renderToast(isError == true ? "error" : "success")}
      {isLoading ? (
        <Box className="App-location-dropdown" sx={{ display: "flex" }}>
          <CircularProgress />
        </Box>
      ) : (
        <></>
      )}
    </div>
  );
}

export default App;
