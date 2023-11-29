import axios, { AxiosRequestConfig } from "axios";

abstract class Endpoint {
  static POST_GENERATOR = 'http://127.0.0.1:8000/api/genconfig/';
  static GET_TXT = 'http://127.0.0.1:8000/api/text/';
  static POST_TXT = 'http://127.0.0.1:8000/api/posttext/';
  static POST_SAM = 'http://127.0.0.1:8000/api/sam/';
}


const DEFAULT_AXIOS_REQUEST_CONFIG: Partial<AxiosRequestConfig> = {
  headers: {
    "Content-Type": "application/json; charset=utf-8",
  },
};

const GET_AXIOS_REQUEST_CONFIG: Partial<AxiosRequestConfig> = {
  headers: {
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Expires': '0',
    "Content-Type": "application/json; charset=utf-8",
  },
};

export interface ApiResponse<T> {
  success: boolean;
  message?: string;
  data?: T;
}
export default class ApiError extends Error {
  constructor(public response: ApiResponse<any>) {
    super(response.message);
  }
}

function call(
  path: Endpoint,
  options: Partial<AxiosRequestConfig>
): Promise<ApiResponse<any>> {
  return axios({
    url: `${String(path)}`,
    ...options,
  })
    .then((response) => {

      return {
        success: true,
        message: undefined,
        data: response.data,
      };
    })
    .catch((error) => {
      if (error.response.statusText.includes("Not Found")) {
        return {
          success: false,
          message: "Not Found"
        }
      }
      return {
        success: false,
        message: error.response.statusText
      }
    });
}


function postToServer<T, R = any>(
  path: Endpoint,
  data?: T,
  options: Partial<AxiosRequestConfig> = DEFAULT_AXIOS_REQUEST_CONFIG,
): Promise<ApiResponse<any>> {
  return call(path, {
    method: "POST",
    data: data,
    ...options,
  });
}

function getFromServer<R = any>(
  path: Endpoint,
  options: Partial<AxiosRequestConfig>
): Promise<ApiResponse<any>> {
  return call(path, {
    method: "GET",
    ...options,
  });
}

export const getText = (): Promise<ApiResponse<String>> =>
  getFromServer(
    Endpoint.GET_TXT,
    {}
  )

export const postText = (data: any): Promise<ApiResponse<any>> =>
  postToServer(
    Endpoint.POST_TXT,
    {
      "data": data
    }
  )

export const postGeneratorConfig = (data: any): Promise<ApiResponse<any>> => 
    postToServer(
      Endpoint.POST_GENERATOR,
      {
        "data": data
      }
    )

export const postSegment = (data: any): Promise<ApiResponse<any>> => {
  console.log(data)
  return postToServer(
    Endpoint.POST_SAM,
    {
      "data": {
        image: data.image,
        size: data.size,
        bbox: data.bbox
      }
    }
  )
}
