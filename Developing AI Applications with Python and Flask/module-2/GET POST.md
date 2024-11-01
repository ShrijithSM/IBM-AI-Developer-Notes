# Request and Response Objects in Flask

## Overview
This guide summarizes the key concepts about Flask's Request and Response objects.

---

## Flask Request Object

### Basic Concepts
- **Request Paths**:
  - Defined with `@app.route` decorator.
  - Defaults to **GET method**.
  - HTTP methods can be customized using `methods` parameter (e.g., `GET`, `POST`).

### Handling HTTP Requests
- For example, `/health` route can handle both GET and POST requests:
  - **GET Request**: Returns `status: ok` and `method: GET`.
  - **POST Request**: Returns `status: ok` and `method: POST`.

### Request Object Attributes
- **Server Address**: Host and port in tuple form.
- **Headers**: Contains request headers like `User-Agent`, `Cache-Control`, etc.
- **URL**: URL path of the requested resource.
- **access_route**: List of IP addresses for forwarded requests.
- **full_path**: Complete request path, including query strings.
- **is_secure**: True if HTTPS/WSS is used.
- **is_JSON**: True if request contains JSON data.
- **Cookies**: Dictionary of cookies sent with the request.

### Common Header Fields
- `Cache-Control`: Specifies caching instructions.
- `Accept`: Specifies content types the client can understand.
- `Accept-Encoding`: Specifies encoding methods the client can decode.
- `User-Agent`: Client information (e.g., browser or OS).
- `Host`: Server hostname and port.

### Customizing Request Object
- Custom request objects are generally optional, as `Flask.Request` has sufficient attributes and methods.

---

## Flask Response Object

### Overview
- A **Response object** is generated to send information back to the client.
- Attributes of the response include:
  - **status_code**: Indicates request success or failure.
  - **headers**: Extra response information.
  - **content_type**: MIME type of response.
  - **content_length**: Size of response body.
  - **content_encoding**: Encoding method of response.
  - **expires**: Expiration date/time for the response.

### Common Response Methods
- `set_cookie`: Sets a cookie in the client's browser.
- `delete_cookie`: Deletes a cookie in the client's browser.

---

## Working with Request and Response Data

### Retrieving Request Data
- **Query Parameters**: Accessible through `args` as a dictionary.
- **JSON Data**: Use `get_JSON()` for JSON parsing.
- **File Uploads**: Accessible via `files`.
- **Form Data**: Accessible via `form`.
- **Combined Values**: Use `values` to combine `args` and `form`.

### Data Parsing Methods
- `get_data()`: Retrieves raw data as bytes from POST request.
- `get_JSON()`: Parses JSON data from POST request.

### Data Extraction Example
- For URL parameters, use:
  - Indexing for strict access (errors if not found).
  - `GET` method for optional access (returns `None` if not found).

### Response Creation Methods
- **Default Response**: Flask automatically creates a 200 response with `mime-type` of HTML.
- **Custom Response**: Use `make_response()` for custom response attributes.
- **Redirect**: Use `redirect()` for 302 status and URL redirection.
- **Error Handling**: `abort()` generates error responses based on conditions.

---

## Summary
- **Flask Request**: Provides data access methods for headers, parameters, cookies, etc.
- **Flask Response**: Allows customizing response attributes and setting headers/cookies.
- **Data Handling**: Multiple methods available for retrieving and parsing request data.
- **Status & Errors**: Use response methods for redirection, setting status codes, and handling errors.
