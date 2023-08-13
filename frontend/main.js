var _a;
function handleClickSubmit() {
    console.log('wassup');
}
function onLogIn() {
    var searchBar = document.querySelector('#ttp-search-bar');
    if (searchBar) {
        searchBar.disabled = false;
        searchBar.placeholder = 'What do you want to know?';
    }
    else {
        console.error('Someone deleted the search bar');
    }
}
var ifr = document.querySelector('iframe[name="utd"]');
console.log(ifr);
var iframe = (_a = ifr === null || ifr === void 0 ? void 0 : ifr.contentDocument) === null || _a === void 0 ? void 0 : _a.body;
// Options for the observer (which mutations to observe)
var config = { childList: true, subtree: true };
// Create an observer instance linked to the callback function
var observer = new MutationObserver(function (mutationList, observer) {
    for (var _i = 0, mutationList_1 = mutationList; _i < mutationList_1.length; _i++) {
        var mutation = mutationList_1[_i];
        if (mutation.type === "childList") {
            console.log('Added a child node');
            var loginBtn = document.querySelector('#btnLoginSubmit');
            if (loginBtn !== null) {
                loginBtn.addEventListener('click', onLogIn);
                observer.disconnect();
            }
        }
    }
});
// Start observing the target node for configured mutations
if (iframe !== null)
    observer.observe(iframe, config);
else
    console.error("iframe not found");
