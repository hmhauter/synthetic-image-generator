import { Box, Button } from "@mui/material";
import Banner from "Banner";
import { Link } from "react-router-dom";

export default function Homepage() {
    return(
        <div>
        <Banner></Banner>
        <Box className="Homepage-button">
        <Link to="/generateImages">
        <Button sx={{
            backgroundColor: "#001965",
            color: "#FCF2F5",
            borderColor: "#282c34",
            ":hover": {
              backgroundColor: "#637099",
              color: "white",
            },
            m: 1,
          }}>Generate Images
          </Button>
          </Link>
          <Link to="/uploadBackground">
          <Button sx={{
            backgroundColor: "#001965",
            color: "#FCF2F5",
            borderColor: "#282c34",
            ":hover": {
              backgroundColor: "#637099",
              color: "white",
            },
            m: 1,
          }}>Upload Background</Button>
          </Link>
          <Link to="/uploadForeground">
          <Button sx={{
            backgroundColor: "#001965",
            color: "#FCF2F5",
            borderColor: "#282c34",
            ":hover": {
              backgroundColor: "#637099",
              color: "white",
            },
            m: 1,
          }}>Upload Foreground</Button>
          </Link>
            <Button sx={{
            backgroundColor: "#001965",
            color: "#FCF2F5",
            borderColor: "#282c34",
            ":hover": {
              backgroundColor: "#637099",
              color: "white",
            },
            m: 1,
          }}>Information</Button>
          </Box>
          </div>
    )
}