import axios from 'axios';

// 1. 기본 인스턴스 생성
const mobiusClient = axios.create({
    baseURL: 'http://localhost:7579', // /Mobius는 리소스 경로(path)에 포함되어 있으므로 제외
    headers: {
        'Accept': 'application/json',
        'X-M2M-RI': '12345',
        'X-M2M-Origin': 'SOrigin',
        'X-M2M-RVI': '3'
    }
});

export default mobiusClient;