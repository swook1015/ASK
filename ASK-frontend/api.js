import axios from 'axios';

const mobiusClient = axios.create({
    baseURL: 'http://localhost:7579', // /Mobius는 리소스 경로(path)에 포함되어 있으므로 제외
    headers: {
        'Accept': 'application/json',
        'X-M2M-RI': '12345',
        'X-M2M-Origin': 'SOrigin',
        'X-M2M-RVI': '3'
    }
});