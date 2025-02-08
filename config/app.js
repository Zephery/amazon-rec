function getBaseURl() {
    if (process.env.NODE_ENV === 'prod') { // 这里改为 process.env.NODE_ENV
        return 'https://rec.wenzhihuai.com/api';
    } else {
        return 'http://localhost:5000';
    }
}

export const HTTP_REQUEST_URL = getBaseURl();
