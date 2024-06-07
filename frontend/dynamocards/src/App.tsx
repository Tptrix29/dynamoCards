import React, {useState} from "react";
import axios from "axios";

const host: string = "http://localhost:8000"

function App() {
  const [youtubelink, setYoutubeLink] = useState<string>("");
  const [responseData, setResponseData] = useState<any>(null);

  const handleLinkChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setYoutubeLink(e.target.value);
  };

  const sendLink = async () => {
    try {
      const response = await axios.post(`${host}/analyze_video`, {
        youtube_link: youtubelink,
      });
      setResponseData(response.data);
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div className="App">
      <h1>Youtube Link to Flashcards Generator</h1>
      <input 
        type="text" 
        placeholder="Enter Youtube Link Here"
        value={youtubelink} 
        onChange={handleLinkChange} />
      <button onClick={sendLink}>Generate Flashcards</button>
      {responseData && (
        <div>
          <h2>Response Data: </h2>
          <p>{JSON.stringify(responseData, null, 2)}</p>
        </div>
      )}
    </div>
  );
}

export default App;