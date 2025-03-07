function getBaseURl() {
    if (process.env.NODE_ENV === 'prod') { // 这里改为 process.env.NODE_ENV
        return 'https://rec.wenzhihuai.com/api';
    } else {
        return 'http://127.0.0.1:5000';
    }
}

export const HTTP_REQUEST_URL = getBaseURl();
