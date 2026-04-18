// Get references to DOM elements
const loginBtn = document.querySelector('.login-btn');
const signupBtn = document.querySelector('.signup-btn');
const modal = document.getElementById('auth-modal');
const closeBtn = document.querySelector('.close-btn');
const dropdownToggles = document.querySelectorAll('.dropdown-toggle');
const authForm = document.getElementById('auth-form');
const authSubmitBtn = document.getElementById('auth-submit-btn');
const toggleAuthLink = document.getElementById('toggle-to-signup');
const googleAuthBtn = document.querySelector('.google-auth-btn');
let isLoginMode = true;

// This function handles a successful authentication by displaying an alert and reloading the page.
function handleSuccess(data) {
    alert(data.message);
    window.location.reload(); 
}

// This function handles switching between Login and Signup modes in the modal.
function toggleAuthMode() {
    isLoginMode = !isLoginMode;
    authSubmitBtn.textContent = isLoginMode ? 'Login' : 'Sign Up';
    const toggleText = isLoginMode ? 'New user?' : 'Already have an account?';
    const linkText = isLoginMode ? 'Sign Up' : 'Login';
    document.querySelector('.toggle-auth').innerHTML = `${toggleText} <span id="toggle-to-signup">${linkText}</span>`;
    document.getElementById('toggle-to-signup').addEventListener('click', toggleAuthMode);
}

// This function opens the authentication modal and sets the correct mode (login or signup).
function openModal(mode){
modal.style.display='flex'; // This activates the centering!
isLoginMode=(mode==='login');
authSubmitBtn.textContent=isLoginMode?'Login':'Sign Up';
const toggleText=isLoginMode?'New user?':'Already have an account?';
const linkText=isLoginMode?'Sign Up':'Login';
document.querySelector('.toggle-auth').innerHTML=`${toggleText} <span id="toggle-to-signup">${linkText}</span>`;
document.getElementById('toggle-to-signup').addEventListener('click',toggleAuthMode);
}

// Event listeners to open the modal
if (loginBtn) {
    loginBtn.addEventListener('click', () => openModal('login'));
}
if (signupBtn) {
    signupBtn.addEventListener('click', () => openModal('signup'));
}

// This closes the modal when the close button is clicked.
closeBtn.addEventListener('click', function() {
    modal.style.display = 'none';
});

// This closes the modal when the user clicks outside of the content.
window.addEventListener('click', function(e) {
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// This manages the dropdown menu behavior.
dropdownToggles.forEach(toggle => {
    toggle.addEventListener('click', function() {
        const parentItem = this.closest('.nav-item');
        parentItem.classList.toggle('show');
    });
});

// This closes all dropdowns when clicking anywhere else on the window.
window.addEventListener('click', function(e) {
    if (!e.target.closest('.dropdown')) {
        document.querySelectorAll('.nav-item.show').forEach(item => {
            item.classList.remove('show');
        });
    }
});

// This attaches the initial toggle listener.
if (toggleAuthLink) {
    toggleAuthLink.addEventListener('click', toggleAuthMode);
}

// This handles the form submission for standard login and signup.
if (authForm) {
    authForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const username = document.getElementById('auth-username').value;
        const password = document.getElementById('auth-password').value;
    
        const endpoint = isLoginMode ? '/login' : '/signup';
    
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username: username, password: password })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && isLoginMode) {
                // If login is successful, call the handleSuccess function.
                handleSuccess(data);
            } else if (data.success && !isLoginMode) {
                // For a successful signup, alert the user and switch to login mode.
                alert(data.message);
                openModal('login');
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error(`Error during ${isLoginMode ? 'login' : 'signup'}:`, error);
            alert('An error occurred. Check the console for details.');
        });
    });
}


// This handles the simulated Google authentication flow.
if (googleAuthBtn) {
    googleAuthBtn.addEventListener('click', function() {
        const googleEmail = prompt("Simulate Google Sign-In: Enter your Google Email:");
    
        if (!googleEmail) {
            alert("Google Sign-In cancelled or no email entered.");
            return;
        }
        
        const tempPassword = 'google_verified_password'; 
        const endpoint = '/google_auth';
    
        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username: googleEmail, password: tempPassword }) 
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // If Google authentication is successful, handle the success with a page reload.
                handleSuccess(data);
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error during Google authentication:', error);
            alert('An error occurred during authentication.');
        });
    });
}
