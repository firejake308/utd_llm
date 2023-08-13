function handleClickSubmit() {
console.log('wassup')

}

function onLogIn() {
    const searchBar = document.querySelector<HTMLInputElement>('#ttp-search-bar');
    if (searchBar) {
      searchBar.disabled = false;
      searchBar.placeholder = 'What do you want to know?';
    }
    else {
      console.error('Someone deleted the search bar')
    }
}

const ifr = document.querySelector<HTMLIFrameElement>('iframe[name="utd"]');
console.log(ifr);
const iframe = ifr?.contentDocument?.body;

// Options for the observer (which mutations to observe)
const config = { childList: true, subtree: true };

// Create an observer instance linked to the callback function
const observer = new MutationObserver((mutationList, observer) => {
  for (const mutation of mutationList) {
    if (mutation.type === "childList") {
      console.log('Added a child node')
        const loginBtn = document.querySelector('#btnLoginSubmit');
        if (loginBtn !== null) {
            loginBtn.addEventListener('click', onLogIn);
            observer.disconnect();
        }
    }
  }
});

// Start observing the target node for configured mutations
if (iframe)
  observer.observe(iframe, config);
else
  console.error("iframe not found");
