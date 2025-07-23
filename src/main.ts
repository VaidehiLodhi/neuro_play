const learning_rate_input = document.getElementById('learning-rate') as HTMLInputElement;
const learning_rate_value = document.getElementById('learning-rate-value') as HTMLSpanElement;

if(learning_rate_input && learning_rate_value) {
  learning_rate_input.addEventListener('input', ()=>{
    learning_rate_value.textContent = learning_rate_input.value;
  });
}

// const likey_text = document.getElementById('likey-text') as HTMLInputElement;
// const likey_text_value = document.getElementById('likey-text-value') as HTMLSpanElement;

// if(likey_text && likey_text_value) {
//   likey_text.addEventListener('input', ()=>{
//     likey_text_value.textContent = likey_text.value;
//   });
// }