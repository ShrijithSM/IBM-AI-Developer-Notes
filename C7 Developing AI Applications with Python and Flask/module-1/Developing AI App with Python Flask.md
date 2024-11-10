# Notes on Application Development Lifecycle

## Overview:
The application development lifecycle (ADLC) is a structured process that all applications follow to go from concept to deployment and maintenance. The lifecycle consists of seven distinct phases, ensuring that every aspect of the application is carefully considered, built, and maintained.

### Seven Phases of the Application Development Lifecycle:

1. **Requirement Gathering**:
   - Collect **user**, **business**, and **technical** requirements.
   - Aim to capture as many requirements as possible, even if they seem minor or redundant.
   - Identify constraints, such as technical limitations or business model challenges.
   - **Example (Hotel Reservation App)**:
     - **User requirement**: Users must be able to view available rooms and amenities.
     - **Business requirement**: The app should determine the correct charge for different rooms and services.
     - **Technical requirement**: The app must work on all browsers and mobile devices.

2. **Analysis**:
   - Analyze the requirements and constraints to design potential solutions.
   - Multiple rounds of verification and revision are often required.
   - Ensure **proper documentation** is maintained to record all updates and changes.

3. **Design**:
   - Design the complete solution for the application.
   - Create clear documentation that specifies how each part of the application should function.
   - The final design is passed on to the development team for coding.

4. **Code and Test**:
   - Develop the application according to the design specifications.
   - Test the code at the programming level (unit testing) to ensure each component meets the requirements.
   - Revise and retest the application as needed.
   - Once all components are working correctly, the team generates an acceptable version of the application.

5. **User and System Test**:
   - **User Testing**: Verify that the application functions as expected from a user's perspective.
   - **System Testing**: Perform tests such as:
     - **Integration Testing**: Ensures that all parts of the application work together.
     - **Performance Testing**: Evaluates the application's speed, scalability, and stability under various workloads.

6. **Production**:
   - The application is made available to end users.
   - It must function properly and remain in a **steady state** where no changes are made unless there is an error.

7. **Maintenance**:
   - Maintenance may involve **fixing errors** or **upgrading the application** with new features.
   - New features must go through all previous phases before being deployed.

### Best Practice: Organizing Code into Multiple Files
- Applications have multiple functionalities, and it’s a best practice to code each functionality in separate files.
- A **central program** should be created to run the application and call the individual files for specific functions.
- **Benefits**:
   - Efficient and easy code maintenance.
   - New functionalities can be added by coding them in separate files, which are then integrated into the main application after testing and verification.

## Summary:
- The seven phases of the application development lifecycle include: **Requirement Gathering**, **Analysis**, **Design**, **Code and Test**, **User and System Test**, **Production**, and **Maintenance**.
- Each phase is essential to ensure the application functions correctly and meets user and business needs.
- Efficient coding practices, such as dividing the application’s functionalities into multiple files, ensure ease of maintenance and scalability.
