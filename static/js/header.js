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

// Auth state
function handleSuccess(data) {
  alert(data.message);
  window.location.reload();
}

function toggleAuthMode() {
  isLoginMode = !isLoginMode;
  authSubmitBtn.textContent = isLoginMode ? 'Login' : 'Sign Up';
  const toggleText = isLoginMode ? 'New user?' : 'Already have an account?';
  const linkText = isLoginMode ? 'Sign Up' : 'Login';
  document.querySelector('.toggle-auth').innerHTML = `${toggleText} <span id="toggle-to-signup">${linkText}</span>`;
  document.getElementById('toggle-to-signup').addEventListener('click', toggleAuthMode);
}

function openModal(mode) {
  modal.style.display = 'flex';
  isLoginMode = (mode === 'login');
  authSubmitBtn.textContent = isLoginMode ? 'Login' : 'Sign Up';
  const toggleText = isLoginMode ? 'New user?' : 'Already have an account?';
  const linkText = isLoginMode ? 'Sign Up' : 'Login';
  document.querySelector('.toggle-auth').innerHTML = `${toggleText} <span id="toggle-to-signup">${linkText}</span>`;
  document.getElementById('toggle-to-signup').addEventListener('click', toggleAuthMode);
}

if (loginBtn) {
  loginBtn.addEventListener('click', () => openModal('login'));
}
if (signupBtn) {
  signupBtn.addEventListener('click', () => openModal('signup'));
}

if (closeBtn && modal) {
  closeBtn.addEventListener('click', function() {
    modal.style.display = 'none';
  });
}

window.addEventListener('click', function(e) {
  if (modal && e.target === modal) {
    modal.style.display = 'none';
  }
});

dropdownToggles.forEach(toggle => {
  toggle.addEventListener('click', function() {
    const parentItem = this.closest('.nav-item');
    parentItem.classList.toggle('show');
  });
});

window.addEventListener('click', function(e) {
  if (!e.target.closest('.dropdown')) {
    document.querySelectorAll('.nav-item.show').forEach(item => {
      item.classList.remove('show');
    });
  }
});

if (toggleAuthLink) {
  toggleAuthLink.addEventListener('click', toggleAuthMode);
}

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
        body: JSON.stringify({
          username: username,
          password: password
        })
      })
      .then(response => response.json())
      .then(data => {
        if (data.success && isLoginMode) {

          handleSuccess(data);
        } else if (data.success && !isLoginMode) {

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
        body: JSON.stringify({
          username: googleEmail,
          password: tempPassword
        })
      })
      .then(response => response.json())
      .then(data => {
        if (data.success) {

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
