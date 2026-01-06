// _________________________________________
// Dynamically creating navbar
// _________________________________________

const nav = document.createElement("nav");

const ul = document.createElement("ul");

const links = [
    { text: "Homepage", href: "/" },
    // {
    //     text: "The projects",
    //     class: "drop-down",
    //     children: [
    //         { text: "Project description", href: "/description" },
    //         { text: "Results", href: "/project" },
    //     ],
    // },

    { text: "Project description", href: "/description" },
    { text: "Experimenting with the models", href: "/project" },
    { text: "Contact", href: "/contact" },
];

links.forEach((link) => {
    const li = document.createElement("li");
    li.classList.add("nav");

    const a = document.createElement("a");

    a.href = link.href;

    a.textContent = link.text;

    li.appendChild(a);
    ul.appendChild(li);

    // if (link.class) {
    //     const div = document.createElement("div");
    //     div.id = "menu";

    //     li.classList.add(link.class);
    //     const ulSub = document.createElement("ul");

    // link.children.forEach((child) => {
    //     const liSub = document.createElement("li");
    //     const aSub = document.createElement("a");
    //     aSub.href = child.href;
    //     aSub.textContent = child.text;

    //     liSub.appendChild(aSub);
    //     ulSub.appendChild(liSub);
    // });

    // const div2 = document.createElement("div");
    // const div3 = document.createElement("div");
    // div3.classList.add("arrow");
    // div2.appendChild(div3);
    // div2.appendChild(ulSub);

    //     const space = document.createElement("div");
    //     space.id = "space";
    //     div.appendChild(space);
    //     div.appendChild(div2);
    //     li.appendChild(div);
    // }
});

nav.appendChild(ul);

document.body.querySelector("header").appendChild(nav);

// _________________________________________
// Drop-down menu
// _________________________________________

const menu = document.getElementsByClassName("drop-down");
Array.from(menu).forEach((el) => {
    el.addEventListener("mouseenter", () => {
        el.classList.add("down");
    });

    el.addEventListener("mouseleave", () => {
        el.classList.remove("down");
    });
});
